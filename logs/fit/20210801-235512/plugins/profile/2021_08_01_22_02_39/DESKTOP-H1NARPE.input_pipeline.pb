$	B(?ye@b??"?r@4Lm?????!???h??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???h??@??B=M9@1?`"?}@I|{נ/%0@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails 4Lm?????i;???.??1-C??6j?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???P1Χ?1????yj?I9?@d?&??r11*	333333E@2T
Iterator::Root::ParallelMapV2?Zd;??!}?	??Q@)?Zd;??1}?	??Q@:Preprocessing2E
Iterator::RootQ?|a2??!`??}iX@)46<?R??1????!?9@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????MbP?!?sHM0?@)????MbP?1?sHM0?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIH?`? @Q???s??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???,? @?a#??5-@!??B=M9@	!       "$	???Ҿc@??2I?q@-C??6j?!?`"?}@*	!       2	!       :	?z????@a??5?"@!|{נ/%0@B	!       J	!       R	!       Z	!       b	!       JGPUb qH?`? @y???s??V@