$	g?x5??f@?F{???s@m7?7M?}?!_~??L<?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'_~??L<?@ظ?]??9@1mU?G?@I,??E|?0@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails m7?7M?}?[???iv?1K?8???\?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!O?)??Y??1??????p?I??<,Ԛ??r11*	?????<u@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??ͪ????!j?r70@J@)????(??1?ֺ?RH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?~j?t???!x??\}@<@)??HP??1???%7F0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatf??a?֤?!ɣKn??'@)?k	??g??15?V@?N&@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatlxz?,C??!<?{?> @)J+???1??<?*?@:Preprocessing2T
Iterator::Root::ParallelMapV2??ZӼ???!??8@)??ZӼ???1??8@:Preprocessing2E
Iterator::Root????o??!p???"@)?0?*??1[8???@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?J?4??!?2z'??@)?J?4??1?2z'??@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipf?c]?F??!?"hO[O@)?ZӼ?}?1n?E9? @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceF%u?k?!Q?v???)F%u?k?1Q?v???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ei?!?̛?1??)a??+ei?1?̛?1??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeǺ???f?!=?M?R^??)Ǻ???f?1=?M?R^??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!=?M?R^??)Ǻ???V?1=?M?R^??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI R?{@Q?*?NHW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[-??B!@0?9??-@!ظ?]??9@	!       "$	!>???2e@+W0[r@K?8???\?!mU?G?@*	!       2	!       :	????b?@sf???#@!,??E|?0@B	!       J	!       R	!       Z	!       b	!       JGPUb q R?{@y?*?NHW@