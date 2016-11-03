import constants as c
import os

import process_data
import avg_runner
import write_frames_and_actions



if __name__ == '__main__':
  write_frames_and_actions.main()
  process_data.main()
  avg_runner.main()


