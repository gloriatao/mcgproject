import os
import sed_eval
import dcase_util

#97
# Start evaluating
file_list = [
    {
     'reference_file': '/home/D/glo/Projects_other/MCG/pred/csv/new/cnn_lstm_all/gt.csv',
     'estimated_file': '/home/D/glo/Projects_other/MCG/pred/csv/new/cnn_lstm_all/pred.csv'

    },
]
CLASSES = ['Q','R','S','T']
data = []

# Get used event labels
all_data = dcase_util.containers.MetaDataContainer()
for file_pair in file_list:
    reference_event_list = sed_eval.io.load_event_list(
        filename=file_pair['reference_file']
    )
    estimated_event_list = sed_eval.io.load_event_list(
        filename=file_pair['estimated_file']
    )

    data.append({'reference_event_list': reference_event_list,
                 'estimated_event_list': estimated_event_list})

    all_data += reference_event_list

event_labels = all_data.unique_event_labels
# Create metrics classes, define parameters
segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
    event_label_list=CLASSES,
    time_resolution=1.0#5.0 /1000
)

event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
    event_label_list=CLASSES,
    t_collar=3.0#0.250
)

# Go through files
for file_pair in data:
    segment_based_metrics.evaluate(
        reference_event_list=file_pair['reference_event_list'],
        estimated_event_list=file_pair['estimated_event_list']
    )

    event_based_metrics.evaluate(
        reference_event_list=file_pair['reference_event_list'],
        estimated_event_list=file_pair['estimated_event_list']
    )

# Get only certain metrics
overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
print("Accuracy:", overall_segment_based_metrics['accuracy']['accuracy'])

# Or print all metrics as reports
print(segment_based_metrics)
print(event_based_metrics)


