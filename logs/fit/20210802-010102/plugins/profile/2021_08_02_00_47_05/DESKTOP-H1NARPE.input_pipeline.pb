$	W???x?d@QE|??!r@??W9҉?!~͑h@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'~͑h@?q4GV?7@1???^??|@IҌE?٩.@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??W9҉?mU?Y??1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!F%u???1???]/Ma?I????ޘ?r11*	?????\u@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???~?:??!L?h??K@)A??ǘ???1?q?.??I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?a??4???!???<@)h??s???1?ٵ?*?.@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatbX9?Ȧ?!;3???	*@)sh??|???1?w?9?H(@:Preprocessing2E
Iterator::RootL7?A`???!???=O#@)M?O???1?!V幣@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?q??????!Ak?<?A@)??Pk?w??1Ƨ ~ND@:Preprocessing2T
Iterator::Root::ParallelMapV2-C??6??!6?????@)-C??6??16?????@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???Q?~?!oS&ۍ@)???Q?~?1oS&ۍ@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipW?/?'??!???)??N@)9??v??z?12?Ym??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?~j?t?h?!???p+??)?~j?t?h?1???p+??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?~j?t?h?!???p+??)?~j?t?h?1???p+??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?[?!?7<?????)_?Q?[?1?7<?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!(???"x??)a2U0*?S?1(???"x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI0?-5T@Q=?.??
W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-???  @$?^???+@!?q4GV?7@	!       "$	??&??Lc@)D???p@rQ-"??[?!???^??|@*	!       2	!       :	?G?3?y@???)??!@!ҌE?٩.@B	!       J	!       R	!       Z	!       b	!       JGPUb q0?-5T@y=?.??
W@