$	???Bf@?:?]?zs@?W?\T??!?(?7?ހ@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?(?7?ހ@KxB???:@1oH???~@I??#??2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?W?\T???AA)Z???1K?8???\?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???????1?[???u?I}A	]??r11*	?????!w@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?Zd;???!??4&2I@)??ʡE??1I??@ՁG@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapԚ?????!??.JJ:@)??H.?!??1?X.??.@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatM?O???!?????%@)???&??15????5$@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatlxz?,C??!|^?.l?@)-C??6??1?.??@:Preprocessing2T
Iterator::Root::ParallelMapV2^K?=???!?_??@)^K?=???1?_??@:Preprocessing2E
Iterator::Rootsh??|???!u??&m&@)??ZӼ???1Q
?O@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip^K?=???!???S??O@)"??u????1w3???@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch? ?	??!+"??? @)? ?	??1+"??? @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice_?Q?k?!?!???e??)_?Q?k?1?!???e??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?~j?t?h?!?;f^???)?~j?t?h?1?;f^???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!E}?a?J??)????Mb`?1E}?a?J??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!?H="?5??)Ǻ???V?1?H="?5??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?gf ?? @Q3?{??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	M???$?!@????.@!KxB???:@	!       "$	??g??d@<????q@K?8???\?!oH???~@*	!       2	!       :	?,??2?@?a?RZY%@!??#??2@B	!       J	!       R	!       Z	!       b	!       JGPUb q?gf ?? @y3?{??V@