$	!$???f@???(?s@?AA)Z???!qht??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'qht??@\!???<@1?D???~@I?3???0@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?AA)Z?????$xC??1?h㈵?d?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!:"ߥ?%??1
?F?c?I???D???r11*	?????dw@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapԚ?????!f????I@)?????K??1y2"t?OH@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??q????!?P9;?,@)??y?)??1?@???M+@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??yǹ?!???DO?:@)?c]?F??1????c)@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?HP???!?I?7?#@)??W?2ġ?1???!??"@:Preprocessing2E
Iterator::RootL7?A`???!?????!@)????<,??1?P^Cy@:Preprocessing2T
Iterator::Root::ParallelMapV2?]K?=??!څ?H(m@)?]K?=??1څ?H(m@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?s?????!?LX?P@)?5?;Nс?1??;?>?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch	?^)ˀ?!??d??@)	?^)ˀ?1??d??@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?~j?t?h?!2T/4ۥ??)?~j?t?h?12T/4ۥ??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!???HE??)??_vOf?1???HE??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!<???h:??)??_?Le?1<???h:??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!?pp?!???)Ǻ???V?1?pp?!???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?B??? @Q???CL?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???7#@N?p???0@!\!???<@	!       "$	? ?s??d@???4??q@
?F?c?!?D???~@*	!       2	!       :	u?r??@??;?U?#@!?3???0@B	!       J	!       R	!       Z	!       b	!       JGPUb q?B??? @y???CL?V@