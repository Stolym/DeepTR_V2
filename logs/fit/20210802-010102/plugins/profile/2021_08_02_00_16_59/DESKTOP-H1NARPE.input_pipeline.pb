$	K?????f@?,f??s@?+?????!C p????@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'C p????@???QP>@1_?;?~@I??R?1?5@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?+??????PۆQ??1??IӠh^?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?ٮ?ˠ?1??!??j?I?tYLl>??r11*	     ?u@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???~?:??!%???K@)s??A???1?ƥ???I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?鷯??!???Jb?9@)k?w??#??1*h?5#,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??ZӼ???!G+?	?a'@)??d?`T??1?c??$@:Preprocessing2T
Iterator::Root::ParallelMapV2a??+e??!?kQŏl@)a??+e??1?kQŏl@:Preprocessing2E
Iterator::RoottF??_??!(]u?&G+@)?+e?X??1}N???!@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat	?^)ː?!??v??@)?]K?=??1<????|@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchlxz?,C|?!蔹????)lxz?,C|?1蔹????:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?O??n??!??y"MN@)?????w?1[T?#???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range{?G?zt?!i%1?1???){?G?zt?1i%1?1???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ei?!?kQŏl??)a??+ei?1?kQŏl??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???f?!?HA?`???)Ǻ???f?1?HA?`???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!?HA?`???)Ǻ???V?1?HA?`???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?*?ؼL#@Q?:?dh?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???6$@׫???1@!???QP>@	!       "$	?Z??5dd@???oèq@??IӠh^?!_?;?~@*	!       2	!       :	?b?e?C@???P?J)@!??R?1?5@B	!       J	!       R	!       Z	!       b	!       JGPUb q?*?ؼL#@y?:?dh?V@