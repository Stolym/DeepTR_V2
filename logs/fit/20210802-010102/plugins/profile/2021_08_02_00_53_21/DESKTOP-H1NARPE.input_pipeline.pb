$	߻?'?f@ʑ{?3?s@yxρ???!n???h>?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'n???h>?@?լ3??=@1?܁?h@I?<c_??3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails yxρ??????o
+??1?mO???^?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?y?Cn???1?3?ۃ`?IF?Swe??r11*	?????|y@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapF????x??!??X?2fH@)???o_??1Ȕ??ocF@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map      ??!??V???>@)??D????1?s?4@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??_vO??!?????/%@)??#?????1?p????#@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???QI??!xN)A?@)-C??6??1и???@:Preprocessing2T
Iterator::Root::ParallelMapV2Zd;?O???!&wMa?@)Zd;?O???1&wMa?@:Preprocessing2E
Iterator::Root??ͪ?զ?!?? ???%@)??_vO??1?????/@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip!?lV}??!?f?.?4M@)?g??s???1?k?d?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchM?O???!??SJ?@)M?O???1??SJ?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice	?^)?p?!ev????)	?^)?p?1ev????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea??+ei?!	???]S??)a??+ei?1	???]S??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!C?D?{???)?~j?t?h?1C?D?{???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??b?!T?kC??)/n??b?1T?kC??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?|??!@Qg?@}?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	E4????#@r_#?1@!?լ3??=@	!       "$	@?????d@?
`{L"r@?mO???^?!?܁?h@*	!       2	!       :	|+?P@?ڪ'3?&@!?<c_??3@B	!       J	!       R	!       Z	!       b	!       JGPUb q?|??!@yg?@}?V@