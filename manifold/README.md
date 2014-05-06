# Code of the thesis work

# How to run the filter (manifold_filter - adjust paths and files as needed)
```bash
INPUT_FILE=/path/to/your/markers/file
OUTPUT_FOLDER=/where/to/save/results
# PARAMETERS.plist is /home/logos_users/roberto/data/PARAMETERS.plist
PARAMETERS_PLIST_FILE=/path/to/your/PARAMETERS.plist
# Raw substacks are in /fast/armonia_b/Brain/V_000_stitched_new/substacks/full/full
SUBSTACKS_FOLDER=/path/to/your/raw/substacks
DEBUG=0
bash split_and_rebuild.sh $INPUT_FILE $OUTPUT_FOLDER $DEBUG
# Wait until slurm jobs completed...
python main_produce_cleaned_marker.py $OUTPUT_FOLDER/patches/ $OUTPUT_FOLDER-res $DEBUG
```

# How to evaluate performances (barykmeans - adjust paths and files as needed)
```
python roberto_split_markers.py $SUBSTACKS_FOLDER $OUTPUT_FOLDER-res/cleaned.marker $OUTPUT_FOLDER-res
for x in `cat gt-substacks.txt`; do (cd $OUTPUT_FOLDER-res/$x && ln -s $PARAMETERS_PLIST_FILE .); done
THRESHOLD=20
for x in `cat test-substacks.txt`; do python eval_perf.py $SUBSTACKS_FOLDER $x $OUTPUT_FOLDER-res --manifold-distance=$THRESHOLD; done
python results_table.py $OUTPUT_FOLDER-res
```
