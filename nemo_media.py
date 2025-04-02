import mujoco
import numpy as np
import math
import time
import jax.numpy as jnp
import jax
import mediapy
from networks.lstm import HIDDEN_SIZE, DEPTH
from brax.io import model
from brax import math  

OBS_SIZE = 334
ACT_SIZE = 24
DT = 0.01

class RenderEnvironment:
    
    def __init__(self, model_path='nemo4b/scene.xml'):
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.mj_model)
        self.dt = self.mj_model.opt.timestep
        
    def render(self, states, camera=None):
        # Create renderer
        renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        frames = []
        
        # Get camera information
        hip_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'l_hip_yaw')
        pelvis_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        
        # Get camera ID for tracking
        camera_id = -1
        if camera == 'track':
            camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, 'track')
        
        # Set up tracking body if track camera is requested
        if camera == 'track' and camera_id >= 0 and pelvis_id >= 0:
            # Set the camera's target body to pelvis
            self.mj_model.cam_targetbodyid[camera_id] = pelvis_id
            
            # Make sure mode is set to tracking
            self.mj_model.cam_mode[camera_id] = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
        
        for state in states:
            # Reset data to the current state
            self.data.qpos = state.qpos
            self.data.qvel = state.qvel
            self.data.ctrl = state.ctrl
            
            # Forward to update all derived quantities
            mujoco.mj_forward(self.mj_model, self.data)
            
            # Update scene with current state data
            renderer.update_scene(self.data, camera=camera)
            
            # Render and collect the frame
            pixels = renderer.render()
            frames.append(pixels)
            
        return frames

def generate_rollout(lstm=True, policy_path='walk_policy15', n_steps=20000, render_every=1):
    print("Starting simulation...")
    
    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path('nemo4b/scene.xml')
    data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = 0.001
    
    # Create environment for rendering 
    env = RenderEnvironment('nemo4b/scene.xml')
    
    # Find the necessary body/site IDs
    pelvis_b_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_back')
    pelvis_f_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'pelvis_front')
    
    # Get sensor locations
    def get_sensor_data(sensor_name):
        sensor_id = mj_model.sensor(sensor_name).id
        sensor_adr = mj_model.sensor_adr[sensor_id]
        sensor_dim = mj_model.sensor_dim[sensor_id]
        return sensor_adr, sensor_dim

    gyro = get_sensor_data("gyro_pelvis")
    vel_p = get_sensor_data("local_linvel_pelvis")
    
    # Function to get observations
    def _get_obs(data1, s_info):
        inv_pelvis_rot = math.quat_inv(data1.xquat[1])
        angvel = data1.sensordata[gyro[0]: gyro[0] + gyro[1]]
        vel = data1.sensordata[vel_p[0]: vel_p[0] + vel_p[1]]

        grav_vec = math.rotate(jnp.array([0, 0, -1]), inv_pelvis_rot)
        position = data1.qpos[7:]
        velocity = data1.qvel[6:]
        phase = s_info["phase"]
        vel_target = s_info["vel_target"]
        angvel_target = s_info["angvel_target"]
        halt = s_info["halt"]
        carry = s_info["lstm_carry"]
        prev_action = s_info["prev_action"]
        cmd = jnp.array([vel_target[0], vel_target[1], angvel_target[0], halt])

        phase_clock = jnp.array([jnp.sin(phase[0]), jnp.cos(phase[0]),
                               jnp.sin(phase[1]), jnp.cos(phase[1])])

        obs = jnp.concatenate([carry, vel,
                             angvel, grav_vec, position, velocity, prev_action, phase_clock, cmd
                             ])
        return obs
    
    # Load policy
    try:
        saved_params = model.load_params(policy_path)
        print(f"Successfully loaded policy from {policy_path}")
    except Exception as e:
        print(f"Error loading policy: {e}")
        raise
    
    # Setup inference function 
    def makeIFN():
        from brax.training.agents.ppo import networks as ppo_networks
        from networks.lstm import make_ppo_networks
        import functools
        from brax.training.acme import running_statistics
        mpn = make_ppo_networks
        network_factory = functools.partial(
            mpn,
            policy_hidden_layer_sizes=(512, 256, 256, 128))
        normalize = lambda x, y: x
        obs_size = OBS_SIZE
        ppo_network = network_factory(
            obs_size, ACT_SIZE, preprocess_observations_fn=normalize
        )
        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        return make_inference_fn
    
    make_inference_fn = makeIFN()
    inference_fn = make_inference_fn(saved_params)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Action conversion function
    joint_limit = jnp.array(mj_model.jnt_range)
    def tanh2Action(action: jnp.ndarray):
        pos_t = action[:ACT_SIZE//2]
        vel_t = action[ACT_SIZE//2:]

        bottom_limit = joint_limit[1:, 0] 
        top_limit = joint_limit[1:, 1]
        vel_sp = vel_t * 10

        pos_sp = pos_t * 1.0

        return jnp.concatenate([pos_sp, vel_sp])
    
    # Initialize state info
    state_info = {
        "halt": 0.,
        "phase": jnp.array([jnp.pi, 0]),
        "vel_target": jnp.array([0.2, 0]),
        "angvel_target": jnp.array([0.]),
        "prev_action": jnp.zeros(ACT_SIZE),
        "lstm_carry": jnp.zeros([HIDDEN_SIZE * DEPTH * 2]),
        "prev_pos": data.xpos[1],
    }
    
    # Set initial pose
    init_qpos = mj_model.keyframe('stand').qpos
    data.qpos = init_qpos
    data.ctrl = np.zeros([ACT_SIZE])
    mujoco.mj_step(mj_model, data)
    
    # Initialize random key
    rng = jax.random.PRNGKey(0)
    
    # Create a rollout list to store states 
    rollout = []
    
    # Store a state struct with all needed properties 
    class StateStruct:
        def __init__(self, qpos, qvel, ctrl):
            self.qpos = qpos
            self.qvel = qvel
            self.ctrl = ctrl
    
    # Capture initial state
    rollout.append(StateStruct(
        np.array(data.qpos), 
        np.array(data.qvel),
        np.array(data.ctrl)
    ))
    
    # Run simulation
    start_time = time.time()
    step_counter = 0
    walk_forward = True
    
    print(f"Starting simulation with {n_steps} steps...")
    for c in range(n_steps):
        # Set movement targets
        if walk_forward:
            state_info["halt"] = 0.0
            state_info["vel_target"] = jnp.array([0.3, 0.0])
            pp1 = data.site_xpos[pelvis_f_id]
            pp2 = data.site_xpos[pelvis_b_id]
            facing_vec = (pp1 - pp2)[0:2]
            facing_vec = facing_vec / jnp.linalg.norm(facing_vec)
            state_info["angvel_target"] = jnp.array([facing_vec[1] * -2])
            
        # Add halt phase
        if (c > 600 and c < 700):
            state_info["halt"] = 1.0
            state_info["phase"] = jnp.array([0, jnp.pi])
            
        # Execute policy at controller frequency
        if c % round(DT / mj_model.opt.timestep) == 0:
            obs = _get_obs(data, state_info)
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(obs, act_rng)
            
            # Extract action and state from policy output
            raw_action = ctrl[2 * HIDDEN_SIZE * DEPTH:]
            act = tanh2Action(state_info["prev_action"])
            data.ctrl = act
            state_info["prev_action"] = raw_action
            state_info["lstm_carry"] = ctrl[:2 * HIDDEN_SIZE * DEPTH]
            
            step_counter += 1
            
        # Update phase
        state_info["phase"] += 2 * jnp.pi * mj_model.opt.timestep / 1.0
        state_info["phase"] = jnp.mod(state_info["phase"], jnp.pi * 2)
        
        # Step physics
        mujoco.mj_step(mj_model, data)
        
        # Store state for rendering at specified frequency
        if c % render_every == 0:
            # Store the current state 
            rollout.append(StateStruct(
                np.array(data.qpos),
                np.array(data.qvel),
                np.array(data.ctrl)
            ))
            
            if c % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {c}/{n_steps} - Time: {elapsed:.2f}s - States: {len(rollout)}")

    print(f"Simulation completed in {time.time() - start_time:.2f}s with {len(rollout)} states")
    
    # Render video 
    print("Rendering simulation to video...")
    render_start_time = time.time()
    
    # Use env.render with camera='track' 
    frames = env.render(rollout[::render_every], camera='track')
    
    # Save video
    try:
        fps = int(1.0 / (mj_model.opt.timestep * render_every))
        print(f"Saving video with {len(frames)} frames at {fps} FPS")
        mediapy.write_video('nemo_simulation.mp4', frames, fps=fps)
        print(f"Video saved successfully to nemo_simulation.mp4")
    except Exception as e:
        print(f"Error saving video: {e}")
        if frames:
            mediapy.write_image("first_frame.png", frames[0])
            print("First frame saved as first_frame.png")

    print(f"Rendering completed in {time.time() - render_start_time:.2f}s")
    return frames

if __name__ == "__main__":
    try:
        print("Starting rollout generation")
        frames = generate_rollout(lstm=True, n_steps=20000, render_every=2)
        print("Simulation complete! Video saved as 'nemo_simulation.mp4'")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
