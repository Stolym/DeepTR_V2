$	;?3?dFe@?G/?lr@??̔??R?!)<hv]?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails')<hv]?@?????<@1?u6???|@I?πz32@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??̔??R?1??̔??R?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?d??7i??Ǻ?????1rQ-"??[?r11*?????x@)      0=2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap:#J{?/??!?ԃt?L@)"?uq??1?@>E0oH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map=
ףp=??!??~?r?:@)ŏ1w-!??1_?X??/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??|гY??!??Ű%@)a2U0*???1??{?W?#@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??q????!??T?@)??q????1??T?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatZd;?O???!˝&?[?@)8??d?`??1?W?ĳ@:Preprocessing2T
Iterator::Root::ParallelMapV2?q??????!8Y?"?:@)?q??????18Y?"?:@:Preprocessing2E
Iterator::Root?:pΈҞ?!Y?"?:P@)V-???1C6]?F+@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?1w-!??!?0n??`P@)?5?;Nс?1\?7X?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?q????o?!8Y?"?:??)?q????o?18Y?"?:??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeF%u?k?!҉b?v??)F%u?k?1҉b?v??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ei?!?1j?????)a??+ei?1?1j?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!?a?p????)?~j?t?X?1?a?p????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??)_?B"@Q?Ԧ?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	w?H?W#@?C$y0@!?????<@	!       "$	m?	Uc@Y????p@??̔??R?!?u6???|@*	!       2	!       :	(jV?D@??{??$@!?πz32@B	!       J	!       R	!       Z	!       b	!       JGPUb q??)_?B"@y?Ԧ?V@