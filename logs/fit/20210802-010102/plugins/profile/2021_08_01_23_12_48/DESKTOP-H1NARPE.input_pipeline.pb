$	??n#6Q@ק??:=c@?PۆQ??!k??^˂u@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'k??^˂u@?	K<?|7@1j?T?s@Id]?Fx@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?W?\T???ܚt["??1??????`?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?PۆQ??1?PۆQ??r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?PۆQ??1?PۆQ??r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!#??fF???:?Y?X??1??IӠh^?r11*	    0t@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?;Nё\??!?3ՑK@)??ZӼ???16??P^CI@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map????z??!e????.;@))\???(??1?W?"1@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat	?^)ˠ?!?&??AO$@)K?=?U??1)?????"@:Preprocessing2E
Iterator::RootX9??v???!?g???1#@)8??d?`??1?FͿڤ@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorM??St$??!???h??@)M??St$??1???h??@:Preprocessing2T
Iterator::Root::ParallelMapV2A??ǘ???!;c?~@)A??ǘ???1;c?~@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatǺ?????!?? ?l?@)A??ǘ???1;c?~@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchŏ1w-!?!??????@)ŏ1w-!?1??????@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????9#??!Y0??"?O@)?ZӼ?}?1{7??˕@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??_?Le?!??-???)??_?Le?1??-???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range/n??b?!?z7?????)/n??b?1?z7?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!??-???)??_?LU?1??-???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?hc'")@Qvɉ?mW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
		@?H?@?4????$@!?	K<?|7@	!       "$	k?????O@/	??i?a@??IӠh^?!j?T?s@*	!       2	!       :	?ȓ?k&???K\)?i??!d]?Fx@B	!       J	!       R	!       Z	!       b	!       JGPUb q?hc'")@yvɉ?mW@