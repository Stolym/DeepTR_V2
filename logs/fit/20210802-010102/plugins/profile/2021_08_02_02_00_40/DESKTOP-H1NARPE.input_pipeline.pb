$	q?1~f@U???`%s@?+?????!?X?є?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?X?є?@???=?x8@1T???>Q~@I??x?5@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?+??????lt?Oq|?1??9̗g?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?x#?ȏ?̶?ֈ`??1?'eRC[?r11*	     ?x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??|?5^??!>xl0?I@)?~j?t???1'?<6H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapNё\?C??!?~8?\?9@)??H.?!??1
2-?}?,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??ڊ?e??!?C^<'@)Q?|a2??1xl0??$@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat^K?=???!??j?GI%@)?ܵ?|У?1?ꑜS?#@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?? ???!???)W?P@)/n????1???;_?@:Preprocessing2T
Iterator::Root::ParallelMapV22??%䃎?!p?($@)2??%䃎?1p?($@:Preprocessing2E
Iterator::Root??Ɯ?!??΀Y@)F%u???1?w???
@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchΈ?????!y?4???@)Έ?????1y?4???@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range"??u??q?!???nZ??)"??u??q?1???nZ??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?l?!??F???)y?&1?l?1??F???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice_?Q?k?!V=?s?p??)_?Q?k?1V=?s?p??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!???????)??_?LU?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI(?w??*!@Q{
?I??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?߄BT @_?Q?>,@?lt?Oq|?!???=?x8@	!       "$	i?U66d@@?ͯ??q@?'eRC[?!T???>Q~@*	!       2	!       :	g?ؠ@??g?ZO(@!??x?5@B	!       J	!       R	!       Z	!       b	!       JGPUb q(?w??*!@y{
?I??V@