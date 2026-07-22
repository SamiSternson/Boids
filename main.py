"""This is where the main loop of the boids simulation is run. Q closes the program, and P pauses the simulation.
It initializes the pygame window, creates a group of boids, and handles user input and drawing.
The main loop updates the boids' positions and velocities based on the flocking rules 
(other than wall avoidance, which is handled in the boid class itself), and draws the boids and their FOVs to the screen.
Every behavior changes the angle of the boid, but I am experimenting with 
changing the speed of the boids if they get too close to each other.

See boid.py for the Boid sprite class itself (drawing, rotation, wall avoidance,
and applying the per-frame heading/speed changes computed here)."""

import pygame as pg
import random as r
import numpy as np
from numba import njit
import boid as b

pg.init()
width, height = 1800, 1000
screen = pg.display.set_mode((width, height))
pg.display.set_caption("Boids Simulation")
clock = pg.time.Clock()
FPS = 60
SPEED=100/FPS #Makes it so that speed is in pixels per second, not pixels per frame
NUM_BOIDS = 100
BOID_SIZE = 10
"""All of the following radii are squared to avoid unnecessary square root calculations in the flocking math.
The scaling factors (15, 6, 3) are arbitrary and can be adjusted to change the behavior of the boids."""
VISION_RADIUS_SQ           = (BOID_SIZE * 15) ** 2 
VISION_RADIUS              = int(VISION_RADIUS_SQ ** 0.5)  
SEPARATION_RADIUS_SQ       = (BOID_SIZE * 6) ** 2
SUPER_SEPARATION_RADIUS_SQ = (BOID_SIZE * 3) ** 2
COLLISION_RADIUS_SQ        = BOID_SIZE ** 2
FOV_HALF_ANGLE             = 135  #half of the total FOV cone

fov_surface = pg.Surface((width, height), pg.SRCALPHA)
"""I asked Claude to optimize the code, and it took out a lot of the math that was previously
done in the Boid class and moved it to a separate function that can be compiled with Numba."""
@njit
def calculate_flock_math(x, y, vx, vy, angles, speeds, num_boids, vision_sq, separation_sq,
                         super_sep_sq, fov_half_deg):
    da_array = np.zeros(num_boids, dtype=np.float64)
    ds_array=np.zeros(num_boids, dtype=np.float64)
    for i in range(num_boids):
        neighbor_x     = 0.0
        neighbor_y     = 0.0
        visible_boids=[]
        vmag_sq = vx[i]*vx[i] + vy[i]*vy[i]
        super_sep=False #True if this boid is in super separation mode, which overrides all other behaviors (except for wall avoidance)
        
        #The boid is constantly trying to reach its base speed
        #(sign-only nudge of 1/SPEED per frame — the size of the gap doesn't matter here,
        #just its direction — so speed drifts back to baseline after a super-separation event)
        if speeds[i]!=SPEED:
            ds_array[i]=(SPEED-speeds[i])/(SPEED*abs(SPEED-speeds[i]))
        for j in range(num_boids):
            if i == j:
                continue

            dx = x[j] - x[i]
            dy = y[j] - y[i]
            sq_dist = dx*dx + dy*dy
            
            sq_dist = max(sq_dist, 0.0001)  # Avoid division by zero

            dot    = vx[i]*dx + vy[i]*dy
            #Standard dot-product FOV cone test: true if j is within fov_half_deg of
            #boid i's heading (dot>=0 handles the boid facing exactly at/behind it)
            in_fov = (vmag_sq == 0 or dot >= 0 or
                      dot*dot < np.cos(np.deg2rad(fov_half_deg)) * vmag_sq * sq_dist)

            if sq_dist < vision_sq and in_fov:
                dist = sq_dist ** 0.5
                if sq_dist < super_sep_sq:
                    super_sep=True
                    #Speeds up if the too-close boid is behind it (dot<0), slows down if it's in front (dot>0)
                    if dot < 0:
                        ds_array[i]+=100/dist
                    elif dot > 0:
                        ds_array[i]-=100/dist
                    #Maxes out the turn strength to make other forces insignificant 
                    turn_strength=100
                    #Uses the cross product to determine which way to turn to avoid the other boid
                    cross = (vx[i] * dy) - (vy[i] * dx)
                    if   cross > 0: da_array[i] += turn_strength
                    elif cross < 0: da_array[i] -= turn_strength
                else:
                    if sq_dist < separation_sq:
                        #Increases the turn strength based on how close the other boid is
                        turn_strength = 400 / dist
                        cross = (vx[i] * dy) - (vy[i] * dx)
                        if   cross > 0: da_array[i] += turn_strength
                        elif cross < 0: da_array[i] -= turn_strength
                    #If the other boid is not in super separation range, it will be used for alignment
                    #The first block of complex if statements is to make sure the boid turns the shortest direction to align with the other boid
                    if (abs(angles[i]-angles[j])>180 and angles[i]<angles[j]):
                        da_array[i]+=((360+angles[i]-angles[j])*((angles[i]-angles[j])/abs(angles[i]-angles[j])))/50
                    elif (abs(angles[i]-angles[j])>180 and angles[i]>angles[j]):
                        da_array[i]+=((360-angles[i]+angles[j])*((angles[i]-angles[j])/abs(angles[i]-angles[j])))/50
                    else:
                        if   angles[i] > angles[j]: da_array[i] -= (angles[i] - angles[j])/50
                        elif angles[i] < angles[j]: da_array[i] += (angles[j] - angles[i])/50
                    #If the other boid is within the vision radius, it will be added to the list of visible boids
                    if 0 < sq_dist < vision_sq:
                        visible_boids.append((j, sq_dist))
                    
        # Sort the visible boids by distance (this will be usefull later when I implement custom distributions for choosing which neighbor to move towards)
        visible_boids=sorted(visible_boids, key=lambda x: x[1])
        num_visible_boids=len(visible_boids)
        if  num_visible_boids> 0 and not super_sep:
            #I am going to experiment with custom distributions for choosing with neighbor to align with at some point
            # probs=[1/2**(i+1) for i in range(num_visible_boids)]
            # new_probs=[probs[i]*(1/sum(probs)) for i in range(num_visible_boids)]
            # neighbor=np.random.choice(np.arange(0, num_visible_boids), p=new_probs)
            neighbor=r.randint(0, num_visible_boids-1)
            neighbor_x=x[visible_boids[neighbor][0]]
            neighbor_y=y[visible_boids[neighbor][0]]
            dx = neighbor_x - x[i]
            dy = neighbor_y - y[i]
            cross = (vx[i] * dy) - (vy[i] * dx)
            #Note: uses the module-level VISION_RADIUS constant directly (not vision_sq)
            #to scale the steer strength — larger vision radius means a gentler pull
            if   cross > 0: da_array[i] -= VISION_RADIUS/(5*((dx**2+dy**2)**0.5))
            else:           da_array[i] += VISION_RADIUS/(5*((dx**2+dy**2)**0.5))

    return da_array, ds_array


boids = pg.sprite.Group()
#Initializes the boids with random positions and adds them to the boids group
for _ in range(NUM_BOIDS):
    boids.add(b.Boid(
        r.randint(VISION_RADIUS, width  - VISION_RADIUS),
        r.randint(VISION_RADIUS, height - VISION_RADIUS),
        boid_size=BOID_SIZE,
        vision_radius=VISION_RADIUS,  
        width=width,
        height=height,
        speed=SPEED
    ))

running   = True
paused    = False
boid_list = boids.sprites()
#Starts main game loop
while running:
    clock.tick(FPS)
    n_boids_off_screen=0
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_q:
                running = False
            elif event.key == pg.K_p:
                paused = not paused

    screen.fill((0, 0, 0))
    # Only print every 60 frames
    # if pg.time.get_ticks() % 1000 < 17:
    #     print(f"FPS: {clock.get_fps():.1f}")

    # --- Simulation update (skipped while paused) ---
    if not paused:
        x_arr     = np.array([b.x          for b in boid_list], dtype=np.float64)
        y_arr     = np.array([b.y          for b in boid_list], dtype=np.float64)
        vx_arr    = np.array([b.velocity.x for b in boid_list], dtype=np.float64)
        vy_arr    = np.array([b.velocity.y for b in boid_list], dtype=np.float64)
        angle_arr = np.array([b.angle      for b in boid_list], dtype=np.float64)
        speed_arr=np.array([b.speed for b in boid_list], dtype=np.float64)
        #Passes the arrays to the Numba-compiled function to calculate the changes in angle and speed for each boids
        da_updates, ds_updates = calculate_flock_math(
            x_arr, y_arr, vx_arr, vy_arr, angle_arr, speed_arr,
            NUM_BOIDS, VISION_RADIUS_SQ, SEPARATION_RADIUS_SQ,
            SUPER_SEPARATION_RADIUS_SQ, FOV_HALF_ANGLE,
        )
        #Passes the updates back to the boids and calls their update method
        for i, boid in enumerate(boid_list):
            boid.da = da_updates[i]
            boid.ds = ds_updates[i]
            boid.update()

    # --- Rendering (runs every frame, even while paused, so the window stays responsive) ---
    # Draw FOV cones behind boids
    fov_surface.fill((0, 0, 0, 0))
    for boid in boid_list:
        boid.draw_fov(fov_surface, VISION_RADIUS_SQ, FOV_HALF_ANGLE)
    screen.blit(fov_surface, (0, 0))

    for boid in boid_list:
        boid.draw(screen)
        if boid.out_of_bounds==True:
            n_boids_off_screen+=1
    if n_boids_off_screen>0:
        print(n_boids_off_screen)

    pg.display.flip()

pg.quit()