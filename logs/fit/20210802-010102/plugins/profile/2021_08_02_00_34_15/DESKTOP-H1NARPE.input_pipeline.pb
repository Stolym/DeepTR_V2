$	a ??`?e@țus@?iT?d??!?1?	:~?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?1?	:~?@#??]??:@1??B:?~@I?Tm7??3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?iT?d???30??&??1?h㈵?d?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!m???Bɬ?1??+,?o?Ig??ͪ?r11*??????w@)      0=2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?A`??"??!??
 ?L@)?|a2U??1<????4I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapV-?????!?ZeQc?8@)>yX?5ͫ?1??ay?,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatn????!n*=AM?$@)?:pΈ??1?\^??2#@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??H?}??!^mOj??@)9??v????1?Bk<2?@:Preprocessing2E
Iterator::Root??d?`T??!???n??"@)j?t???1???;=?@:Preprocessing2T
Iterator::Root::ParallelMapV2???QI??!??uB7V@)???QI??1??uB7V@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice????????!?,u??
@)????????1?,u??
@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?St$????!	?멛@)?St$????1	?멛@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipGr?????!?֝?mP@)y?&1?|?1S???I???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?~j?t?h?!$???u??)?~j?t?h?1$???u??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!U!o????)Ǻ???f?1U!o????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!?¾?5???)/n??R?1?¾?5???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??I??!@Q"?'??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	qY?? ?!@???W?.@!#??]??:@	!       "$	2?"?9d@?=@??[q@?h㈵?d?!??B:?~@*	!       2	!       :	SH??@	Ԕ??&@!?Tm7??3@B	!       J	!       R	!       Z	!       b	!       JGPUb q??I??!@y"?'??V@