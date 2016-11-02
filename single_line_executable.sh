# FIRST, generate data...

curr_dir=$(pwd)
echo curr dir is $curr_dir
num_train_examples=1000
num_test_exampels=100

echo "Generating frames for ten training games..."
# python GYM/write_frames_and_actions.py -d $curr_dir/Data/Pong_01/Train -n 3

echo "Generating frames for three test games..."
# python GYM/write_frames_and_actions.py -d $curr_dir/Data/Pong_01/Test -n 1

echo "Processing training games..."
# python Code/process_data.py -n $num_train_examples -t $curr_dir/Data/Pong_01/Train -c $curr_dir/Data/Pong_01/.Clips/
echo "Completed data processing..."
echo "Begin training..."
python Code/avg_runner.py