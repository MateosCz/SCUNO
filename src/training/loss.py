import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(7, 8, 9))
def ssm_dsm_loss(params, state, xs, times, x0, Sigmas, drifts, object_fn='Heng', with_x0=True, x_L=12):
    dt = times[1] - times[0]
    # dimensions:
    # Sigmas: (batch_size, num_timesteps, num_landmarks, dim)
    # xs: (batch_size, num_timesteps, num_landmarks, dim)
    # times: (num_timesteps,)
    # x0: (batch_size, num_landmarks, dim)
    # drifts: (batch_size, num_timesteps, num_landmarks, dim)

    # the outer vmap is over the timesteps first then over batch size
    # the inner vmap is over the batch size
    loss = jax.vmap(batched_single_step_loss, in_axes=(None, #params
                                                        None, #state
                                                        1, #x_prev
                                                        1, #x
                                                        0, #t
                                                        None, #x0
                                                        1, #Sigma
                                                        1, #Sigma_prev
                                                        1, #drift_prev
                                                        None, #dt
                                                        None, #object_fn
                                                        None, #with_x0
                                                        None, #x_L
                                                        ))(params, 
                                                               state, 
                                                               xs[:, :-1, ...], 
                                                               xs[:, 1:, ...], 
                                                               times[:-1], 
                                                               x0, 
                                                               Sigmas[:, 1:, ...], 
                                                               Sigmas[:, :-1, ...], 
                                                               drifts[:, :-1, ...], 
                                                               dt, 
                                                               object_fn,
                                                               with_x0,
                                                               x_L)
    
    if object_fn == 'Heng':
        loss = jnp.mean(loss, axis=1)
        loss = jnp.sum(loss) * dt/2
    elif object_fn == 'Yang':
        loss = jnp.mean(loss, axis=1)
        loss = jnp.sum(loss) * dt/2

    return loss
        
def single_step_loss(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn='Heng', with_x0=True, x_L=12):
    print("x.shape", x.shape)
    print("x_prev.shape", x_prev.shape)
    print("drift_prev.shape", drift_prev.shape)
    print("dt.shape", dt.shape)
        
    if object_fn == 'Heng':
        if with_x0:
            pred_score = state.apply_fn(params, x, t, x0, x_L)
        else:
            pred_score = state.apply_fn(params, x, t, x_L)

        # check the x's shape, if it is a 2D manifold data, then we need to flatten it
        if x.ndim == 3:
            x = jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
            x_prev = jnp.reshape(x_prev, (x_prev.shape[0] * x_prev.shape[1], x_prev.shape[2]))
            pred_score = jnp.reshape(pred_score, (pred_score.shape[0] * pred_score.shape[1], pred_score.shape[2]))
            drift_prev = jnp.reshape(drift_prev, (drift_prev.shape[0] * drift_prev.shape[1], drift_prev.shape[2]))
            Sigma_prev = jnp.reshape(Sigma_prev, (Sigma_prev.shape[0] * Sigma_prev.shape[1], Sigma_prev.shape[2] * Sigma_prev.shape[3]))
            Sigma = jnp.reshape(Sigma, (Sigma.shape[0] * Sigma.shape[1], Sigma.shape[2] * Sigma.shape[3]))
            Sigma_prev = Sigma_prev + 1e-3 * jnp.eye(Sigma_prev.shape[0])
            Sigma_prev_inv = jnp.linalg.solve(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))
            g_approx = -jnp.matmul(Sigma_prev_inv, (x - x_prev - dt * drift_prev))/dt
            
            diff = pred_score - g_approx
            loss = jnp.linalg.norm(jnp.matmul(diff.T, jnp.matmul(Sigma * dt, diff))) ** 2
            return loss

        Sigma_prev = Sigma_prev + 1e-3 * jnp.eye(Sigma_prev.shape[0])
        Sigma_prev_inv = jnp.linalg.solve(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))
        g_approx = -jnp.matmul(Sigma_prev_inv, (x - x_prev - dt * drift_prev))/dt
        
        diff = pred_score - g_approx
        loss = jnp.linalg.norm(jnp.matmul(diff.T, jnp.matmul(Sigma * dt, diff))) ** 2
    elif object_fn == 'Yang':
        if with_x0:
            pred_score = state.apply_fn(params, x, t, x0, x_L)
        else:
            pred_score = state.apply_fn(params, x, t, x_L)

        b = -(x - x_prev - dt * drift_prev) / dt

        loss = jnp.mean(jnp.sum(jnp.square(pred_score - b), axis=-1),axis=-1)


    return loss

# vmap over batch size, one batch's loss is mean at each timestep's loss
def batched_single_step_loss(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn='Heng', with_x0=True, x_L=12):
    batched_loss = jax.vmap(single_step_loss, in_axes=(None, #params
                                                        None, #state
                                                        0, #x_prev
                                                        0, #x
                                                        None, #t
                                                        0, #x0
                                                        0, #Sigma
                                                        0, #Sigma_prev
                                                        0, #drift_prev
                                                        None, #dt
                                                        None, #object_fn
                                                        None, #with_x0
                                                        None, #x_L
                                                        ))(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn, with_x0, x_L)
    return batched_loss

