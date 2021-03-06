import numpy as np
import getopt
import sys
from glob import glob
import os

import constants as c
from utils import process_clip


def actions_to_one_hot(actions):
    one_hot_action = np.zeros((len(actions), c.NUM_POSSIBLE_MOVES), dtype=np.float32)
    for num, act in enumerate(actions):
        one_hot_action[num][int(act)] = 1.0
    return one_hot_action

def actions_to_one_hot_layers(height, width, actions):
    one_hot_layers_actions = np.zeros((height, width, len(actions)*c.NUM_POSSIBLE_MOVES))
    # I think this is right, print to make sure.
    for num, act in enumerate(actions):
        one_hot_layers_actions[:,:,(num*c.NUM_POSSIBLE_MOVES + int(act))] = 1.0
    return one_hot_layers_actions

def actions_and_clips_to_layered_turns(actions, clips):
    height, width, num_clip_channels = clips.shape
    total_hist_length = len(actions)
    num_possible_actions = c.NUM_POSSIBLE_MOVES
    
    total_num_channels = num_clip_channels + (total_hist_length*num_possible_actions)
    total_output = np.zeros((height, width, total_num_channels), dtype=np.float32)

    num_channels_per_turn = 3 + num_possible_actions

    # Starts out all zeros. Copy the picture channels in. Then,
    # since it's layer-actions, the first channel of the layer is (turn*num_channels_per_turn)
    # Then you go three forward for the RGB, then you go forward until the right action,
    # and then you set this to one.
    for turn in xrange(total_hist_length):
        total_output[:,:,turn*num_channels_per_turn:turn*num_channels_per_turn+3] \
            = clips[:,:,turn*3:(turn+1)*3]
        total_output[:,:,(turn*num_channels_per_turn) + 3 + int(actions[turn])] = 1.0

    return total_output



def process_training_data(num_clips):
    """
    Processes random training clips from the full training data. Saves to TRAIN_DIR_CLIPS by
    default.

    @param num_clips: The number of clips to process. Default = 5000000 (set in __main__).

    @warning: This can take a couple of hours to complete with large numbers of clips.
    # clip: A clip of some size, that is probably 5 frames long. Shape: (1,height, width, 3*5).
    # actions: a list of actions for each frame, which is probably 5 frames long. Shape: (1, 5)
    # one_hot_actions: a list of actions for each frame, one-hotted. Shape: (1, num_possible_moves, 5)
    # one_hot_layers: a list of actions for each frame, one-hot-layered. Shape: (1, height, width, 5*num_possible_moves)
    # AND THEN, TO COMBINE THEM: You need to have it sort of stack correctly, 
    # So it should go 1_R, 1_G, 1_B, 1_A1, 1_A2, ...2_R,2_G...
    """

    num_prev_clips = len(glob(c.TRAIN_DIR_CLIPS + '*'))


    for clip_num in xrange(num_prev_clips, num_clips + num_prev_clips):
        clip, actions = process_clip()
        height, width, num_clip_channels = clip.shape
        total_hist_length = len(actions)
        if num_clip_channels != 3*total_hist_length:
            print('should be 3 times as many clip channels as actions')
            raise Exception('should be 3 times as many clip channels as actions')
        # CLIP SHAPE IS FROM ONE EXAMPLE, NOT MORE!



        one_hot_action = actions_to_one_hot(actions)
        one_hot_layers_actions = actions_to_one_hot_layers(height, width, actions)
        total_output = actions_and_clips_to_layered_turns(actions, clip)
            
        if (clip_num + 1) % 100 == 0: print 'Processed %d clips' % (clip_num + 1)

        np.savez_compressed(c.TRAIN_DIR_CLIPS + str(clip_num), 
            clip=clip,
            actions=actions,
            one_hot_action=one_hot_action,
            one_hot_layers_actions=one_hot_layers_actions,
            concat_clip_actions=total_output)




def usage():
    print 'Options:'
    print '-n/--num_clips= <# clips to process for training> (Default = 5000000)'
    print '-t/--train_dir= <Directory of full training frames>'
    print '-c/--clips_dir= <Save directory for processed clips>'
    print "                (I suggest making this a hidden dir so the filesystem doesn't freeze"
    print "                 with so many files. DON'T `ls` THIS DIRECTORY!)"
    print '-o/--overwrite  (Overwrites the previous data in clips_dir)'
    print '-H/--help       (Prints usage)'


def main():
    ##
    # Handle command line input
    ##

    num_clips = c.NUM_CLIPS_TO_PROCESS

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'n:t:c:oH',
                                ['num_clips=', 'train_dir=', 'clips_dir=', 'overwrite', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-n', '--num_clips'):
            num_clips = int(arg)
        if opt in ('-t', '--train_dir'):
            c.TRAIN_DIR = c.get_dir(arg)
        if opt in ('-c', '--clips_dir'):
            c.TRAIN_DIR_CLIPS = c.get_dir(arg)
        if opt in ('-o', '--overwrite'):
            c.clear_dir(c.TRAIN_DIR_CLIPS)
        if opt in ('-H', '--help'):
            usage()
            sys.exit(2)

    # set train frame dimensions
    assert os.path.exists(c.TRAIN_DIR)
    c.FULL_HEIGHT, c.FULL_WIDTH = c.get_train_frame_dims()

    ##
    # Process data for training
    ##
    process_training_data(num_clips)


if __name__ == '__main__':
    main()
