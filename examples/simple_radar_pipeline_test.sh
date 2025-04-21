#!/bin/bash

COMMAND_PARAMETERS=("../build/examples/simple_radar_pipeline" "../build/examples/simple_radar_pipeline_stf")
STREAM_PARAMETERS=("1" "2" "4" "5" "6")
#SAMPLE_PARAMETERS=("--numSamples 1000" "--numSamples 4500" "--numSamples 9000")
SAMPLE_PARAMETERS=("1000" "2000" "4500" "6000" "10000")

NUM_RUNS=5

OUTPUT_FILE="radar_heatmap_data.csv"
# Initialize the CSV file

# Loop through the parameters
for str_param in "${STREAM_PARAMETERS[@]}"; do
  OUTPUT_FILE="radar_heatmap_data_${sam_param}.csv"
  echo "Command,NumStreams,NumSamples,AverageGbps" > $OUTPUT_FILE
  for command_param in "${COMMAND_PARAMETERS[@]}"; do
    for sam_param in "${SAMPLE_PARAMETERS[@]}"; do
      TOTAL_GBPS=0
      for i in $(seq 1 $NUM_RUNS); do
        #echo "Iteration $i with parameter $command_param $str_param $sam_param"
        OUTPUT=$($command_param --numStreams $str_param --numSamples $sam_param)
        GBPS=$(echo "$OUTPUT" | grep -oP '(?<=\().*? Gbps' | awk '{print $1}')

        # Add the extracted value to the total
        if [ -n "$GBPS" ]; then
          TOTAL_GBPS=$(echo "$TOTAL_GBPS + $GBPS" | bc)
        else
          echo "Failed to extract Gbps for iteration $i."
        fi
      done

      # Calculate the average
      if [ "$NUM_RUNS" -gt 0 ]; then
        AVERAGE_GBPS=$(echo "$TOTAL_GBPS / $NUM_RUNS" | bc -l)
        #echo "$command_param $str_param $sam_param verage Gbps over $NUM_RUNS runs: $AVERAGE_GBPS"
        # Append the results to the CSV file
        echo "$command_param,$(echo $str_param | awk '{print $0}'),$(echo $sam_param | awk '{print $0}'),$AVERAGE_GBPS"
        echo "$command_param,$(echo $str_param | awk '{print $0}'),$(echo $sam_param | awk '{print $0}'),$AVERAGE_GBPS" >> $OUTPUT_FILE
      else
        echo "No runs were performed."
      fi
    done
  done
done

echo "Heatmap data saved to $OUTPUT_FILE."
