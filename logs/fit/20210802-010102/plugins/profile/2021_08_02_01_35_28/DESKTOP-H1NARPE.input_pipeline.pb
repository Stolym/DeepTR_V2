$	??S??
h@%??N??t@fL?g?Q?!Л?T??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'Л?T??@?|?F?>@1????s?@IJ???3@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsfL?g?Q?1fL?g?Q?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?W??I???n??Ř?1?E?n?1?r11*	?????}@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?s?????!??XK??I@)2w-!???1????BUH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??镲??!?W?Q?<@)??\m????1o4u~?/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeath??|?5??!t???#c)@)?v??/??1s??`Ԇ(@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???H??!??Q?]@)h??|?5??1t???#c@:Preprocessing2T
Iterator::Root::ParallelMapV2?I+???!?????@)?I+???1?????@:Preprocessing2E
Iterator::Root䃞ͪϥ?!C??U?T"@)?0?*??1?XK?a?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??"??~??!??!?O@)?St$????1$I?$I?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?&S???!W??FS??)?&S???1W??FS??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice???_vOn?!?M?+y??)???_vOn?1?M?+y??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!?D?f???)HP?s?b?1?D?f???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????Mb`?!???????)????Mb`?1???????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6Z?!?`???)-C??6Z?1?`???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??XE??!@QH?T?a?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:]??$@???3{?1@!?|?F?>@	!       "$	??????e@\4Zh1?r@fL?g?Q?!????s?@*	!       2	!       :	?b?0-@C??f?&@!J???3@B	!       J	!       R	!       Z	!       b	!       JGPUb q??XE??!@yH?T?a?V@