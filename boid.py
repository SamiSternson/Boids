"""
boid.py

Defines the Boid sprite: how an individual boid is drawn, rotated, moved,
and kept off the edges of the screen. The flocking rules that make boids
move as a group (separation, alignment, cohesion, "super separation") are
computed elsewhere, in main.py's calculate_flock_math function, and handed
to each Boid every frame as self.da (change in heading, degrees) and
self.ds (change in speed). Wall avoidance is the one behavior that lives
entirely inside this class, since it only depends on a boid's own position
and heading, not on its neighbors.
"""

import numpy as np
import pygame as pg
import random as r
import math


class Boid(pg.sprite.Sprite):
    """A single triangular boid sprite.

    Each Boid tracks its own position, heading, and speed, and is
    responsible for drawing itself, rotating its sprite to match its
    heading, steering away from the screen edges, and advancing its
    position each frame. Flocking behavior (separation/alignment/cohesion)
    is computed externally by main.py and applied via self.da / self.ds
    before update() is called each frame.
    """

    def __init__(self, x, y, boid_size, vision_radius, width, height, speed):
        super().__init__()
        self.x = x
        self.y = y
        self.speed = speed
        self.base_speed = speed  # anchor value; speed is clamped to [0.5x, 1.5x] this and drifts back toward it (see update)
        self.color = (r.randint(100, 255), r.randint(100, 255), r.randint(100, 255)) 
        self.original_surface = pg.Surface((boid_size, boid_size), pg.SRCALPHA)
        pg.draw.polygon(self.original_surface, self.color,
                        ((0, boid_size), (boid_size//2, 0), (boid_size, boid_size)))
        self.display = self.original_surface

        self.rect = self.display.get_rect(center=(self.x, self.y))
        self.angle = r.randint(0, 360)
        self.da = 0
        self.ds = 0
        self.velocity = pg.Vector2(0, -1)

        self.boid_size = boid_size
        self.vision_radius = vision_radius  
        self.width = width
        self.height = height

        self.left   = pg.Rect(0,0, self.vision_radius,height)
        self.right  = pg.Rect(width - self.vision_radius,0, self.vision_radius,height)
        self.top    = pg.Rect(0,0, width,self.vision_radius)
        self.bottom = pg.Rect(0, height - self.vision_radius, width,self.vision_radius)
        self.random_offset = 0    # small jitter added to the wall-turn velocity thresholds; re-rolled only while clear of every wall (see avoid_walls)
        self.out_of_bounds=False  # sticky debug flag — set True the first time this boid strays off-screen, and never reset back
    def draw(self, surface):
        """Blit the boid's current (rotated) sprite onto surface."""
        #Uncomment these lines to draw the wall detection rectangles for debugging

        # pg.draw.rect(surface, (255, 0, 0), self.top, 1)
        # pg.draw.rect(surface, (0, 255, 0), self.bottom, 1)
        # pg.draw.rect(surface, (0, 0, 255), self.left, 1)
        # pg.draw.rect(surface, (255, 255, 0), self.right, 1)

        surface.blit(self.display, self.rect)

    def draw_fov(self, surface, VISION_RADIUS_SQ, FOV_HALF_ANGLE):
        """Draw a translucent cone onto surface showing this boid's field of view.

        Approximates the FOV as a filled fan spanning FOV_HALF_ANGLE
        degrees on either side of the boid's current heading, out to a
        radius of sqrt(VISION_RADIUS_SQ). Purely cosmetic — it mirrors, but
        has no effect on, the actual vision/FOV cone test used in
        calculate_flock_math.
        """
        fwd  = math.atan2(-self.velocity.y, self.velocity.x)
        half = math.radians(FOV_HALF_ANGLE)

        points = [(self.x, self.y)]
        steps  = 28
        for k in range(steps + 1):
            a = fwd - half + (2 * half * k / steps)
            points.append((
                self.x + (VISION_RADIUS_SQ**0.5) * math.cos(a),
                self.y - (VISION_RADIUS_SQ**0.5) * math.sin(a),
            ))

        r_col, g_col, b_col = self.color
        pg.draw.polygon(surface, (r_col, g_col, b_col, 20), points)

    def rotate(self):
        """Apply the accumulated heading change (self.da) and regenerate
        the rotated sprite/rect to match the new angle."""
        self.angle = (self.angle + self.da) % 360
        self.display = pg.transform.rotate(self.original_surface, self.angle)
        self.rect = self.display.get_rect(center=(self.x, self.y))

    def avoid_walls(self):
        """Steer the boid away from the screen edges.

        Runs every frame before self.da (already set by the flocking math
        in main.py) is applied by update(). Two layers of response:

        1. Panic override — within `critical_dist` (3 boid-lengths) of any
           edge, self.da is reset to 0, discarding whatever the flocking
           rules wanted this frame, so escaping the wall takes priority
           over everything else.
        2. Graded steering — for each edge the boid's rect currently
           overlaps, any da left over from step 1 is damped in proportion
           to how close the boid is to that edge (`da *= dist / (vr*5)`,
           so it fades out approaching the wall), then a turn-toward-open-
           space term is added on top. If the boid is heading almost
           straight at the wall (within ~15 degrees of the normal) it
           commits to a hard left/right turn based on which half of the
           screen it's on; otherwise it turns whichever way is already the
           shorter way back toward the middle.

        The four wall checks below are independent `if`s rather than
        `elif`s, so a boid in a corner can be steered by two walls in the
        same frame.
        """
        vr = self.vision_radius
        critical_dist = self.boid_size*3 
        turn_scaling = 5
        #In order to add some randomness to the wall avoidance, the boid will have a random offset that is added to the threshold for turning away from the wall.
        if not (self.rect.colliderect(self.left)  or self.rect.colliderect(self.right) or
                self.rect.colliderect(self.top)   or self.rect.colliderect(self.bottom)):
            self.random_offset = r.uniform(-0.1, 0.1)
        else:
            #Inside this distance from the wall, the boid will ignore all other forces and just try to turn away from the wall
            if (self.y<critical_dist or self.y>self.height-critical_dist or
                self.x<critical_dist or self.x>self.width-critical_dist):
                self.da = 0 
        
        #The following code block handles the wall avoidance behavior of the boid. 
        #It checks which wall the boid is near and adjusts the boid's angle accordingly to avoid collision.
        #If the angle between the boid's heading vector and the normal vector of the wall is less than 15 degrees, the boid will turn towards the middle of the screen. Otherwise, it will turn away from the wall in the direction that requires the least amount of turning.
        # --- Top wall ---
        if self.rect.colliderect(self.top) and self.velocity.y < 0.15 + self.random_offset:
            if self.angle <= 15 or self.angle >= 345:
                self.da=self.da*(self.y/(vr*5))
                
                if self.x < self.width/2: 
                    self.da -= turn_scaling * vr / max(self.y, 1)
                else:                     
                    self.da += turn_scaling * vr / max(self.y, 1)
            elif self.angle > 0 and self.angle < 180: self.da += turn_scaling * vr / max(self.y, 1)
            else: self.da -= turn_scaling 

        # --- Bottom wall ---
        if self.rect.colliderect(self.bottom) and self.velocity.y > -0.15 + self.random_offset:
            dist = self.height - self.y
            self.da=self.da*(dist/(vr*5))
            if self.angle >= 165 and self.angle <= 195:
                if self.x < self.width/2: 
                    self.da += turn_scaling * vr / max(dist, 1)
                else:                     
                    self.da -= turn_scaling * vr / max(dist, 1)
            elif self.angle > 180: self.da += turn_scaling * vr / max(dist, 1)
            else: self.da -= turn_scaling 

        # --- Left wall ---
        if self.rect.colliderect(self.left) and self.velocity.x < 0.15 + self.random_offset:
            self.da=self.da*(self.x/(vr*5)  )
            if self.angle <= 105 and self.angle >= 75:
                if self.y < self.height/2: 
                    self.da += turn_scaling * vr / max(self.x, 1)
                else:                      
                    self.da -= turn_scaling * vr / max(self.x, 1)
            elif self.angle > 90 and self.angle < 180: self.da += turn_scaling * vr / max(self.x, 1)
            else: self.da -= turn_scaling 
        # --- Right wall ---
        if self.rect.colliderect(self.right) and self.velocity.x > (-0.15 + self.random_offset):
            dist = self.width - self.x
            self.da=self.da*(dist/(vr*5))
            if self.angle >= 255 and self.angle <= 285:
                if self.y < self.height/2: 
                    self.da -= turn_scaling * vr / max(dist, 1)
                else:                      
                    self.da += turn_scaling * vr / max(dist, 1)
            elif 90 < self.angle < 270: self.da -= turn_scaling * vr / max(dist, 1)
            else: self.da += turn_scaling * vr / max(dist, 1)
        
        
    def update(self):
        """Advance the boid by one frame.

        Expects self.da and self.ds to already be set by the caller (the
        main loop assigns them from calculate_flock_math's output before
        calling update()). Order of operations: run wall avoidance (which
        may override da), clamp the turn rate, apply and clamp the speed
        change, rotate the sprite, then move.
        """
        if self.x<0 or self.x>self.width or self.y<0 or self.y>self.height:
            self.out_of_bounds=True
        self.avoid_walls()
        #Maxes out da at 15 degrees per frame to prevent the boid from turning too quickly
        if self.da>0:
            self.da=min(self.da, 15)
        elif self.da<0:
            self.da=max(self.da, -15)
        #ds (set externally from calculate_flock_math) is added to speed every frame, then
        #clamped to [0.5x, 1.5x] base_speed — this is what lets a boid speed up/slow down
        #during a super-separation event and drift back to normal afterward
        self.speed+=self.ds
        self.speed=max(self.base_speed/2, min(self.speed, self.base_speed*1.5))
        self.rotate()
        self.velocity = pg.Vector2(0, -self.speed)
        self.velocity.rotate_ip(-self.angle)

        self.x += self.velocity.x
        self.y += self.velocity.y
        self.rect.center = (self.x, self.y)