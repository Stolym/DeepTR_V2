$	?kAo?U@?m<??e@?`???!?E{?P?u@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?E{?P?u@?*????;@1a?^C??s@IT8?T?]@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?7?0???b?o???1vk???i?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?`???1?`???r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!CV?zNz??1ŭ???g?Ig+/?????r11*	43333st@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??_vO??!?z???gJ@)??#?????1feтH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?a??4???!D]????=@)?^)?Ǫ?1'????/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat,e?X??!_?kt6?+@)?=yX???1u??&?)@:Preprocessing2T
Iterator::Root::ParallelMapV2?0?*???!E???v?@)?0?*???1E???v?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatjM????!U8???Y@)?q??????1u??&@:Preprocessing2E
Iterator::Root?W[?????!?4? ?u"@)M?O???1?_??±@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch????Mb??!(?U?@)????Mb??1(?U?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipx$(~???!:DP?N@)? ?	??1(?U??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?l?!%#k??)y?&1?l?1%#k??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice-C??6j?!o~?f?K??)-C??6j?1o~?f?K??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeǺ???f?!????Db??)Ǻ???f?1????Db??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!???L?x??)a2U0*?S?1???L?x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??N??"@Q?*???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	p???@"p?ۅ?+@!?*????;@	!       "$	?p)?S@لaX??c@ŭ???g?!a?^C??s@*	!       2	!       :	??t???????TlO@!T8?T?]@B	!       J	!       R	!       Z	!       b	!       JGPUb q??N??"@y?*???V@