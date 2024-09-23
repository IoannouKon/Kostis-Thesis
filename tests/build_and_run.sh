#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <source-file.c>"
  exit 1
fi

# Set the source file and output names
SOURCE_FILE=$1
OBJECT_FILE="${SOURCE_FILE%.c}.o"
OUTPUT_BINARY="${SOURCE_FILE%.c}.riscv"

# Compile the source file into an object file
riscv64-unknown-elf-gcc -fno-common -fno-builtin-printf -specs=htif_nano.specs -c "$SOURCE_FILE" -o "$OBJECT_FILE"

# Check if the compilation was successful
if [ $? -ne 0 ]; then
  echo "Compilation failed"
  exit 1
fi

# Link the object file into an executable binary
riscv64-unknown-elf-gcc -static -specs=htif_nano.specs "$OBJECT_FILE" -o "$OUTPUT_BINARY" -lm -u_printf_float -lc 

# Check if the linking was successful
if [ $? -ne 0 ]; then
  echo "Linking failed"
  exit 1
fi

# Inform the user that the process is complete
echo "Binary created: $OUTPUT_BINARY"

# Run the binary with Spike
spike "$OUTPUT_BINARY"

# Remove the object file
rm -f "$OBJECT_FILE"

echo "Removed object file: $OBJECT_FILE"

