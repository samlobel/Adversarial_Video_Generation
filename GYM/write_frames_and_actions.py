import gym
from time import sleep
env = gym.make('Pong-v0')
from PIL import Image
import os
import json
import argparse
# It's going to make a config file, that has num_frames, num_actions,

# http://stackoverflow.com/questions/4711880/pil-using-fromarray-with-binary-data-and-writing-coloured-text
# http://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
def write_observation(pic_arr, frame_num, ep_number, data_dir, prefix='Frame'):
  subdir_name = os.path.join(data_dir, str(ep_number).zfill(5))
  if not os.path.isdir(subdir_name):
    os.makedirs(subdir_name)
  im = Image.fromarray(pic_arr).convert('RGB')
  filename = os.path.join(subdir_name, prefix + str(frame_num).zfill(5) + '.png')
  im.save(filename)

# def write_config(num_moves, data_dir, frame_prefix='Frame', action_prefix='Action'):
#   if not os.path.isdir(data_dir):
#     os.makedirs(data_dir)
#   conf = {
#     'num_moves' : num_moves,
#     'frame_prefix' : frame_prefix,
#     'action_prefix' : action_prefix
#   }
#   json_conf = json.dumps(conf)
#   with open(os.path.join(data_dir, 'config.json'),'w') as f:
#     f.write(json_conf)
#   print('conf written')

def write_action(action, frame_num, ep_number, data_dir, prefix='Action'):
  subdir_name = os.path.join(data_dir, str(ep_number).zfill(5))
  if not os.path.isdir(subdir_name):
    os.makedirs(subdir_name)
  filename = os.path.join(subdir_name, prefix + str(frame_num).zfill(5) + '.txt')

  with open(filename, 'w') as f:
    f.write(str(action))


# WOW, I had that WRONG! I was writing the action and the resulting observation,
# when I should have written the action and the observation itself.

def main(data_dir, num_games=10):
  num_games = int(num_games)
  num_actions = env.action_space.n
  print('num actions is ' + str(num_actions))
  reward = 0
  # write_config(num_actions, data_dir)
  for game_num in range(num_games):
    print('starting game {}'.format(game_num))
    observation = env.reset()
    frame_num = 0
    while True:
      action = env.action_space.sample()
      write_action(action, frame_num, game_num, data_dir)
      write_observation(observation, frame_num, game_num, data_dir)
      observation, reward, done, info = env.step(action)
      if done:
        print('Episode finished after {} timesteps'.format(frame_num+1))
        observation = env.reset()
        break
      frame_num += 1

def return_parsed_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data_dir')
  parser.add_argument('-n', '--num_games')
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = return_parsed_args()
  print('data dir: ' + str(args.data_dir))
  print('num games: ' + str(args.num_games))
  main(args.data_dir, num_games=args.num_games)

