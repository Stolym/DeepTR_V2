$	??????f@?|??t?s@-C??6??!A???8(?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'A???8(?@?????(9@1c????@I?E?~E2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails -C??6???[?????1???]/Ma?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!fl?f???1K?8???l?Ia?$?Ӣ?r11*	ffffff6@2T
Iterator::Root::ParallelMapV2???????!J?$I??I@)???????1J?$I??I@:Preprocessing2E
Iterator::Rootj?t???!n۶m??W@)??ׁsF??1?$I?$F@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!%I?$I@)??H?}M?1%I?$I@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???㏬@Q?B?7W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	O4"+?? @?Dq?-@!?????(9@	!       "$	???e@h?(?x>r@???]/Ma?!c????@*	!       2	!       :	aO;?5i@ ?VP?%@!?E?~E2@B	!       J	!       R	!       Z	!       b	!       JGPUb q???㏬@y?B?7W@