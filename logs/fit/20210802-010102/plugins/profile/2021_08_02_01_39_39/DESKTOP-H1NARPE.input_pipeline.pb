$	?SN??e@
??	?r@t^c??ފ?!?h???_?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?h???_?@??rJ@d<@1??ME??}@Iٯ;?y?1@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails t^c??ފ?}?;l"3??1?_>Y1\]?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??ϷK??1???מYb?I???8???r11*	fffff?x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???S????!E?©iPK@)?A?f???1?1	??G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??St$???!??N?z5?@)<?R?!???16?/?}K5@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatn????!?=D??#@)㥛? ???1'bZv"@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch????????!??tJ@)????????1??tJ@:Preprocessing2E
Iterator::RootV-???!?{??^V@)???Q???1????XY@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?Q?????!j7??@)?<,Ԛ???1??6*?@:Preprocessing2T
Iterator::Root::ParallelMapV2y?&1???!wXdS@)y?&1???1wXdS@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??v????!.???v?N@)??~j?t??1?-5?8@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??H?}m?!9?5ג"??)??H?}m?19?5ג"??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!?b$?Kx??)?????g?1?b$?Kx??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!kh!????)??_vOf?1kh!????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!&???al??)a2U0*?S?1&???al??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?M???!@Q?]?J??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??*o?"@???f}c0@!??rJ@d<@	!       "$	q?Q?c@?δמ<q@?_>Y1\]?!??ME??}@*	!       2	!       :		Q??@9m?4a?$@!ٯ;?y?1@B	!       J	!       R	!       Z	!       b	!       JGPUb q?M???!@y?]?J??V@