$	???W?g@?B?w?s@t^c??ފ?!ظ?]_N?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'ظ?]_N?@T?{F"H=@1V(???l@IK?8???5@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails t^c??ފ?3j?J>v??1?'eRC[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!oӟ?H??1ŭ???g?I??|????r11*	?????w@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapjM????!<6V?`*M@)??m4????1??zH?(J@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?0?*???!????_?5@)?a??4???1jgǏuc*@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???~?:??!{<?kJ-!@)%u???11F$}?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat]m???{??!ƈ䣏#@)0*??D??1؛?? ?@:Preprocessing2T
Iterator::Root::ParallelMapV28??d?`??!?p~??@)8??d?`??1?p~??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+e??!??K?M?
@)a??+e??1??K?M?
@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceZd;?O???!??94??@)Zd;?O???1??94??@:Preprocessing2E
Iterator::Root???B?i??!ؠ?? @)n????1!?m??=@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchHP?sׂ?!)?aý?@)HP?sׂ?1)?aý?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?~j?t???!h?????Q@)ŏ1w-!?1?p???x @:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP?s?b?!)?aý???)HP?s?b?1)?aý???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!V???[??)?~j?t?X?1V???[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?<?y1o"@Qh????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(.?`?#@i?Vn?0@!T?{F"H=@	!       "$	?3g}J?d@ڟ?ͮ$r@?'eRC[?!V(???l@*	!       2	!       :	???Y@gRSk )@!K?8???5@B	!       J	!       R	!       Z	!       b	!       JGPUb q?<?y1o"@yh????V@