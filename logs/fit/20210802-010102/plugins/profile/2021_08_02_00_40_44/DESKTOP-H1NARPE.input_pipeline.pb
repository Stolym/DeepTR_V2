$	??ڊ??f@?o??s@a2U0*???!i?-X0?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'i?-X0?@r???D8@1??Gp??@I3??(H1@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails a2U0*???D??<???1?mO???^?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!s???$??1??I???r?I5)?^??r11*	?????iy@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?	h"lx??!?[?.nI@)??????1??O??9E@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?????̼?!%??? ?;@)?QI??&??1	k???o1@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??_?L??!8@"?]v$@)?j+??ݣ?1o???#@:Preprocessing2T
Iterator::Root::ParallelMapV2?W[?????!?Dϲ??@)?W[?????1?Dϲ??@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatz6?>W??!?????D@)??+e???1.? ???@:Preprocessing2E
Iterator::Root?	h"lx??!?[?.n)@)j?t???1?S?{?&@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchjM????! ?.;?@)jM????1 ?.;?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??Pk?w??!$?eGY@)??Pk?w??1$?eGY@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?sF????!fGY???M@)??y?):??1??OՂ@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeǺ???f?!?l?xQ	??)Ǻ???f?1?l?xQ	??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!??gj???)a2U0*?c?1??gj???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!??gj???)a2U0*?S?1??gj???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?Ag?+@Q苉D?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?I??/ @?p?"?,@!r???D8@	!       "$	dGЅ?/e@??vH:Yr@?mO???^?!??Gp??@*	!       2	!       :	]??a??@yll0?M#@!3??(H1@B	!       J	!       R	!       Z	!       b	!       JGPUb q?Ag?+@y苉D?W@