$	?I???g@j?Yj??s@fL?g?a?!?? ??L?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?? ??L?@?Lۿ?n<@1?%W?8?@I+1?JZ?3@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsfL?g?a?1fL?g?a?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!????g???1???מYr?I3??3???r11*	??????D@2T
Iterator::Root::ParallelMapV2䃞ͪϕ?!?؉?ؙI@)䃞ͪϕ?1?؉?ؙI@:Preprocessing2E
Iterator::Rootf??a?֤?!؉?؉uX@)?j+??ݓ?1;?;QG@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!??N??N@)??H?}M?1??N??N@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??zDh!@Q??p???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,3?*w?"@?zT?[j0@!?Lۿ?n<@	!       "$	q\?M?e@
?:? =r@fL?g?a?!?%W?8?@*	!       2	!       :	?JY?8V@??Y?&@!+1?JZ?3@B	!       J	!       R	!       Z	!       b	!       JGPUb q??zDh!@y??p???V@