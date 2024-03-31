#!/bin/sh

# Define the source file and target directory
SOURCE_FILE="racetrack_latest.pt"
TARGET_DIR="checkpoints"

# Check if the target directory exists, create if not
if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p "$TARGET_DIR"
fi

# Loop from 1 to 10 to copy and rename the file
for n in {1..60}
do
  echo "number: "
  echo "$n"
  python train_ddpg.py
  cp "$SOURCE_FILE" "$TARGET_DIR/racetrack_latest_${n}.pt"
done

echo "Files have been copied and renamed successfully."
