$	?u??we@?? ?r@????Ŋ??!/m8,??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'/m8,??@:#/k2<@1e??~#>}@IR?h!3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ????Ŋ??H?Sȕz??1!>???@`?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???t ???1? 3??O\?I?]?V$&??r11*	gffff?E@2T
Iterator::Root::ParallelMapV246<?R??!;?K?W,I@)46<?R??1;?K?W,I@:Preprocessing2E
Iterator::Root?g??s???!P??zX@)?0?*??1fC?}??G@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!??`Р @)??H?}M?1??`Р @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI [?1b"@Q??ݹ?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	w^='?"@Z*??F0@!:#/k2<@	!       "$	??w?~c@? ?e?p@? 3??O\?!e??~#>}@*	!       2	!       :	?b2Be?@??+?&@!R?h!3@B	!       J	!       R	!       Z	!       b	!       JGPUb q [?1b"@y??ݹ?V@