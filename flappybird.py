#! /usr/bin/env python3

"""Flappy Bird, implemented using Pygame."""

import math
import os
import sys
from random import randint
from collections import deque

import pygame
from pygame.locals import *
import pickle
import numpy as np
import random


FPS = 60
ANIMATION_SPEED = 0.18  # pixels per millisecond
WIN_WIDTH = 284 * 2     # BG image size: 284x512 px; tiled twice
WIN_HEIGHT = 512


class Bird(pygame.sprite.Sprite):
    """Represents the bird controlled by the player.

    The bird is the 'hero' of this game.  The player can make it climb
    (ascend quickly), otherwise it sinks (descends more slowly).  It must
    pass through the space in between pipes (for every pipe passed, one
    point is scored); if it crashes into a pipe, the game ends.

    Attributes:
    x: The bird's X coordinate.
    y: The bird's Y coordinate.
    msec_to_climb: The number of milliseconds left to climb, where a
        complete climb lasts Bird.CLIMB_DURATION milliseconds.

    Constants:
    WIDTH: The width, in pixels, of the bird's image.
    HEIGHT: The height, in pixels, of the bird's image.
    SINK_SPEED: With which speed, in pixels per millisecond, the bird
        descends in one second while not climbing.
    CLIMB_SPEED: With which speed, in pixels per millisecond, the bird
        ascends in one second while climbing, on average.  See also the
        Bird.update docstring.
    CLIMB_DURATION: The number of milliseconds it takes the bird to
        execute a complete climb.
    """

    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.18
    CLIMB_SPEED = 0.3
    CLIMB_DURATION = 333.3

    def __init__(self, x, y, msec_to_climb, images):
        """Initialise a new Bird instance.

        Arguments:
        x: The bird's initial X coordinate.
        y: The bird's initial Y coordinate.
        msec_to_climb: The number of milliseconds left to climb, where a
            complete climb lasts Bird.CLIMB_DURATION milliseconds.  Use
            this if you want the bird to make a (small?) climb at the
            very beginning of the game.
        images: A tuple containing the images used by this bird.  It
            must contain the following images, in the following order:
                0. image of the bird with its wing pointing upward
                1. image of the bird with its wing pointing downward
        """
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, delta_frames=1):
        """Update the bird's position.

        This function uses the cosine function to achieve a smooth climb:
        In the first and last few frames, the bird climbs very little, in the
        middle of the climb, it climbs a lot.
        One complete climb lasts CLIMB_DURATION milliseconds, during which
        the bird ascends with an average speed of CLIMB_SPEED px/ms.
        This Bird's msec_to_climb attribute will automatically be
        decreased accordingly if it was > 0 when this method was called.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        if self.msec_to_climb > 0:
            frac_climb_done = 1 - self.msec_to_climb/Bird.CLIMB_DURATION
            self.y -= (Bird.CLIMB_SPEED * frames_to_msec(delta_frames) *
                       (1 - math.cos(frac_climb_done * math.pi)))
            self.msec_to_climb -= frames_to_msec(delta_frames)
        else:
            self.y += Bird.SINK_SPEED * frames_to_msec(delta_frames)

    @property
    def image(self):
        """Get a Surface containing this bird's image.

        This will decide whether to return an image where the bird's
        visible wing is pointing upward or where it is pointing downward
        based on pygame.time.get_ticks().  This will animate the flapping
        bird, even though pygame doesn't support animated GIFs.
        """
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        else:
            return self._img_wingdown

    @property
    def mask(self):
        """Get a bitmask for use in collision detection.

        The bitmask excludes all pixels in self.image with a
        transparency greater than 127."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        """Get the bird's position, width, and height, as a pygame.Rect."""
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    """Represents an obstacle.

    A PipePair has a top and a bottom pipe, and only between them can
    the bird pass -- if it collides with either part, the game is over.

    Attributes:
    x: The PipePair's X position.  This is a float, to make movement
        smoother.  Note that there is no y attribute, as it will only
        ever be 0.
    image: A pygame.Surface which can be blitted to the display surface
        to display the PipePair.
    mask: A bitmask which excludes all pixels in self.image with a
        transparency greater than 127.  This can be used for collision
        detection.
    top_pieces: The number of pieces, including the end piece, in the
        top pipe.
    bottom_pieces: The number of pieces, including the end piece, in
        the bottom pipe.

    Constants:
    WIDTH: The width, in pixels, of a pipe piece.  Because a pipe is
        only one piece wide, this is also the width of a PipePair's
        image.
    PIECE_HEIGHT: The height, in pixels, of a pipe piece.
    ADD_INTERVAL: The interval, in milliseconds, in between adding new
        pipes.
    """

    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img, pipe_body_img):
        """Initialises a new random PipePair.

        The new PipePair will automatically be assigned an x attribute of
        float(WIN_WIDTH - 1).

        Arguments:
        pipe_end_img: The image to use to represent a pipe's end piece.
        pipe_body_img: The image to use to represent one horizontal slice
            of a pipe's body.
        """
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()   # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -                  # fill window from top to bottom
             3 * Bird.HEIGHT -             # make room for bird to fit through
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT          # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

        self.top_y = top_pipe_end_y
        self.bottom_y = bottom_pipe_end_y

    @property
    def top_height_px(self):
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):
        """Update the PipePair's position.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)

    def collides_with(self, bird):
        """Get whether the bird collides with a pipe in this PipePair.

        Arguments:
        bird: The Bird which should be tested for collision with this
            PipePair.
        """
        return pygame.sprite.collide_mask(self, bird)


def load_images():
    """Load all images required by the game and return a dict of them.

    The returned dict has the following keys:
    background: The game's background image.
    bird-wingup: An image of the bird with its wing pointing upward.
        Use this and bird-wingdown to create a flapping bird.
    bird-wingdown: An image of the bird with its wing pointing downward.
        Use this and bird-wingup to create a flapping bird.
    pipe-end: An image of a pipe's end piece (the slightly wider bit).
        Use this and pipe-body to make pipes.
    pipe-body: An image of a slice of a pipe's body.  Use this and
        pipe-body to make pipes.
    """

    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folder
        (./images/).  All images are converted before being returned to
        speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        file_name = os.path.join('.', 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            # images for animating the flapping bird -- animated GIFs are
            # not supported in pygame
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png')}


def frames_to_msec(frames, fps=FPS):
    """Convert frames to milliseconds at the specified framerate.

    Arguments:
    frames: How many frames to convert to milliseconds.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return 1000.0 * frames / fps


def msec_to_frames(milliseconds, fps=FPS):
    """Convert milliseconds to frames at the specified framerate.

    Arguments:
    milliseconds: How many milliseconds to convert to frames.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return fps * milliseconds / 1000.0


class Neural_Network(object):
    def __init__(self, weights_list = None, chromosome = None):
        self.fitness = 1 #how fit it is to reproduce (based on the final score in the game)

        self.input_layer_size = 4
        self.output_layer_size = 1
        self.hidden_layer_size = 6

        if weights_list:
            self.W1 = weights_list[0]
            self.W2 = weights_list[1]
        else:
            self.W1 = np.random.randn(self.input_layer_size,self.hidden_layer_size)
            self.W2 = np.random.randn(self.hidden_layer_size,self.output_layer_size)

        self.weights = [self.W1,self.W2]

        if chromosome:
            self.decode_chromosome(chromosome)





        self.get_chromosome()

    def forward(self,X):
        #propagate inputs through network
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        result = self.activation(self.z3)
        return result > .5

    def activation(self,z):
        return 1/(1+np.exp(-z))

    def get_weight(self,n):
        return self.weights[n]

    def get_fitness(self):
        return self.fitness

    def get_chromosome(self):
        self.chromosome = []
        for weight_list in self.weights:
            for row in weight_list:
                for weight in row:
                    self.chromosome.append(weight)
    def decode_chromosome(self,c):
        chromo = c[::-1]
        new_weights = []
        for weight_list in self.weights:
            new_weight_list = []
            for row in weight_list:
                new_row = []
                for item in row:
                    weight = chromo.pop()
                    new_row.append(weight)
                new_weight_list.append(new_row)
            new_weights.append(new_weight_list)
        self.weights = new_weights
        self.W1 = self.weights[0]
        self.W2 = self.weights[1]

def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w > r:
         return c
      upto += w
   assert False, "Shouldn't get here"

def crossover(NN1,NN2, crossover_rate = .7):
    c1 = NN1.chromosome
    c2 = NN2.chromosome
    if random.random() < crossover_rate:
        position = random.randint(0, len(c1)-1)
        new_c1 = c1[:position] + c2[position:]
        new_c2 = c2[:position] + c1[position:]
        return Neural_Network(chromosome=new_c1), Neural_Network(chromosome=new_c2)
    return NN1,NN2

def mutate(NN, mutation_rate = .2):
    c = NN.chromosome
    new_chromo = []
    for i in c:
        if random.random() < mutation_rate:
            a = i + np.random.uniform(-1,1) * .3
            new_chromo.append(a)
        else:
            new_chromo.append(i)
    return Neural_Network(chromosome = new_chromo)

def create_next_gen(gen,population):
    next_gen = []
    individual_chance = [[i,i.get_fitness()] for i in gen]
    for _ in xrange(population/2):
        parent1 = weighted_choice(individual_chance)
        parent2 = weighted_choice(individual_chance)
        parent1,parent2 = crossover(parent1,parent2)
        parent1 = mutate(parent1)
        parent2 = mutate(parent2)
        next_gen += [parent1,parent2]
    return next_gen
"""
def reproduce(first,second):
    #generate two child Neural Networks with a combination of the weights of the two parent NNs
    children = []
    for child in xrange(2): #repeat for however many children you want per couple
        weights_list = [] #stores the new W1 and W2 for the child
        for weight in xrange(2): #goes through W1 and W2 of each parent
            new_weight_list = []
            weight_first = first.get_weight(weight)
            weight_second = second.get_weight(weight)
            for i,row in enumerate(weight_first): #each row
                new_row = []
                for j,first_weight in enumerate(row): #each weight in the row
                    second_weight = weight_second[i][j] #get the weight of the other parent at the same position
                    new_weight = random.choice([second_weight,first_weight])
                    #new_weight = np.random.uniform(first_weight,second_weight) #create a new weight which is a random number between the two parent weights
                    if random.random() < 0: #mutation rate
                        new_weight += random.uniform(-1,1)
                    new_row.append(new_weight)
                new_weight_list.append(new_row)
            weights_list.append(new_weight_list)
        children.append(Neural_Network(weights_list))
    return children

def create_next_gen(gen,population):
    next_gen = []
    individual_chance = [[i,i.get_fitness()] for i in gen]
    for _ in xrange(population/2):
        first = weighted_choice(individual_chance)
        second = weighted_choice(individual_chance)
        next_gen += reproduce(first,second)
    return next_gen
"""

pygame.init()

display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption('Pygame Flappy Bird')

population = 10
gen = [Neural_Network() for _ in xrange(population)]
i = 0
while 1:
    if i:
        print "AVERAGE: " + str( sum([j.fitness for j in gen])/population)
        pickle.dump(gen, open("good_gen.p", "wb"))
        gen = create_next_gen(gen,population)
    i+=1
    print "\n\nGENERATION: " + str(i)
    for x,NN in enumerate(gen):
        print "SPECIES: {0}".format(str(x+1)),
        clock = pygame.time.Clock()
        score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
        images = load_images()

        # the bird stays in the same x position, so bird.x is a constant
        # center bird on screen
        bird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
                    (images['bird-wingup'], images['bird-wingdown']))

        pipes = deque()

        frame_clock = 0  # this counter is only incremented if the game isn't paused
        score = 0
        done = paused = False
        while not done:
            clock.tick(FPS)

            # Handle this 'manually'.  If we used pygame.time.set_timer(),
            # pipe addition would be messed up when paused.
            if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
                pp = PipePair(images['pipe-end'], images['pipe-body'])
                pipes.append(pp)

                #bird.msec_to_climb = Bird.CLIMB_DURATION

            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                    done = True
                    pygame.quit()
                    sys.exit()
                elif e.type == KEYUP and e.key in (K_PAUSE, K_p):
                    paused = not paused
                elif e.type == MOUSEBUTTONUP or (e.type == KEYUP and
                        e.key in (K_UP, K_RETURN, K_SPACE)):
                    bird.msec_to_climb = Bird.CLIMB_DURATION
            if frame_clock % (FPS/2)  == 0:
                X = [[pipes[0].top_y - bird.y, bird.y - pipes[0].bottom_y, pipes[0].x, bird.y]]
                #X = [[(pipes[0].top_height_px + pipes[0].bottom_height_px)/2. - bird.y, pipes[0].x]]
                network_choice = NN.forward(X)
                if network_choice:
                    bird.msec_to_climb = Bird.CLIMB_DURATION
            if paused:
                continue  # don't draw anything

            # check for collisions
            pipe_collision = any(p.collides_with(bird) for p in pipes)
            if pipe_collision or 0 >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                done = True

            for x in (0, WIN_WIDTH / 2):
                display_surface.blit(images['background'], (x, 0))

            while pipes and not pipes[0].visible:
                pipes.popleft()

            for p in pipes:
                p.update()
                display_surface.blit(p.image, p.rect)

            bird.update()
            display_surface.blit(bird.image, bird.rect)

            # update and display score
            for p in pipes:
                if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                    score += 1
                    p.score_counted = True

            score_surface = score_font.render(str(score), True, (255, 255, 255))
            score_x = WIN_WIDTH/2 - score_surface.get_width()/2
            display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

            pygame.display.flip()
            frame_clock += 1


        """
         [
            [-1.77687748  0.05011382  1.16981091]
            [-0.08402794  1.25750322 -0.16428337]
         ]

         [
             [-0.3592772 ]
             [ 0.50106525]
             [ 1.20453393]
         ]"""
        NN.fitness = frame_clock
        #NN.fitness = 1/(math.fabs((pipes[0].top_height_px + pipes[0].bottom_height_px)/2. - bird.y)) + frame_clock/10

        print NN.fitness
        #print('Game over! Score: %i' % score)

#[3.293138612102414, -2.8686126154787841, -1.1659035151569088, 2.629875769169447, 0.17925671509167163, -2.4646330223071917, -0.41657362113415813, 0.86317042456389959, -0.36652423120620259]