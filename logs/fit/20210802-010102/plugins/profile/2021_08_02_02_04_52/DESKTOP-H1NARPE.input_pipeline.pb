$	?????$f@?Aj??,s@?`???!?s???@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?s???@?8
?>@1??1?s~@I??J??2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?`?????Z?a/t?1X?|[?Tg?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!ZI+?????1??????p?IP)?????r11*	effffz@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapQk?w????!3@x???I@)??e??a??1?b?,̹G@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat6<?R???!?5??xp0@)L7?A`???1??l%?/@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapOjM???!??Jy,?<@)}гY????1?,]?g?(@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??Ɯ?!?;5?y?@)p_?Q??1??Gd?@:Preprocessing2T
Iterator::Root::ParallelMapV2tF??_??!??Wg??@)tF??_??1??Wg??@:Preprocessing2E
Iterator::Root?,C????!,(??d%@)??A?f??1\8??9@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchF%u???!#b??D	@)F%u???1#b??D	@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??1??%??!???!?/N@)?St$????1??5?&???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?~j?t?h?!N? .????)?~j?t?h?1N? .????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!z
?ܿ???)??_vOf?1z
?ܿ???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!?^M??`??)a2U0*?c?1?^M??`??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!?C???p??)Ǻ???V?1?C???p??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIH\*H?"@Qw???֧V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?	???$@??????1@!?8
?>@	!       "$	?yV?
d@0???`q@X?|[?Tg?!??1?s~@*	!       2	!       :	+???R/@L?t???%@!??J??2@B	!       J	!       R	!       Z	!       b	!       JGPUb qH\*H?"@yw???֧V@