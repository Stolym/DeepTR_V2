$	Ů?"V?e@????V?r@?d??7i??!y??Mq6?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'y??Mq6?@?oH??:@1U??-%?}@I'?y?3=1@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?d??7i????q?@H??1?3?ۃ`?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???M?q??1]??a??1?3?ۃ`?r11*	     @B@2T
Iterator::Root::ParallelMapV20*??D??!?s?Ν;P@)0*??D??1?s?Ν;P@:Preprocessing2E
Iterator::Rootr??????!??Ǐ?X@)g??j+???1p???@@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip/n??R?!??@)/n??R?1??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIľF? @Q?|'(W?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??eX?!@ t??C?.@??q?@H??!?oH??:@	!       "$	[??v??c@??a?#q@?3?ۃ`?!U??-%?}@*	!       2	!       :	??L?D?@?2??#@!'?y?3=1@B	!       J	!       R	!       Z	!       b	!       JGPUb qľF? @y?|'(W?V@