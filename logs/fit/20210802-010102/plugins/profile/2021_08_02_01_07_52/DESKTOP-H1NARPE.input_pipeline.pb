$	!M??|\f@?Mj?Vs@t	??????!,????@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails',????@>#??=@1=)?Zu~@I???
G?2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails t	??????M??????1???W?h?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?/????????????1??[X7?]?r11*	????̼w@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?3??7??!??`?h?H@)?>W[????1?uB?WE@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map,Ԛ????!? g??Z9@)u????1??ߌ?}/@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatI.?!????!?̶A?&@)Y?8??m??1????%@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat㥛? ???!?y?|8#@)?? ?rh??1)9	w?!@:Preprocessing2T
Iterator::Root::ParallelMapV2?q??????!Ԕ? n @)?q??????1Ԕ? n @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicevq?-??!} ??@)vq?-??1} ??@:Preprocessing2E
Iterator::Root?ׁsF???!6\?9'@)?{??Pk??1??,@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??ZӼ???!d=q(|@)??ZӼ???1d=q(|@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??T?????!?X??O@)?<,Ԛ?}?1???o????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!?n?*?F??)?~j?t?h?1?n?*?F??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range{?G?zd?!d?O??){?G?zd?1d?O??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!??M?;???)/n??R?1??M?;???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???Yb"@Q?"ߴ?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	}??LX@$@???1@M??????!>#??=@	!       "$	?=??INd@U??'ȕq@??[X7?]?!=)?Zu~@*	!       2	!       :	X? ?E@@?m??%@!???
G?2@B	!       J	!       R	!       Z	!       b	!       JGPUb q???Yb"@y?"ߴ?V@