$	??^?Bg@????$t@????G??!?aor?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?aor?@???I?:@1M?J??@IgE?D??6@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ????G?????߆?1?'eRC[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!? 3??O???~j?t???1$Di?]?r11*	??????y@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?߾?3??!?1]?H@)x??#????1?x?^??F@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??????!t?ݚ?2.@)h??|?5??1??6k`?,@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map=?U?????!?1?C1]7@)?Y??ڊ??1?Y??+@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?ܵ?|У?!?	.?S?"@)e?X???1.Ņ??? @:Preprocessing2T
Iterator::Root::ParallelMapV2??????!V?????@)??????1V?????@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch ?o_Ή?!q?ec1n@) ?o_Ή?1q?ec1n@:Preprocessing2E
Iterator::Root<?R?!???!??t?<h$@)Ǻ?????1?VZXH?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?Y??ڊ??!?I??P@)?ZӼ?}?1??rK`???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range	?^)?p?!?G??`???)	?^)?p?1?G??`???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice_?Q?k?!R?m??^??)_?Q?k?1R?m??^??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?k?!_xj????)F%u?k?1_xj????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!??S?0*??)??_?LU?1??S?0*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??????!@Qc`))?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	ך??%?!@5??<??.@???߆?!???I?:@	!       "$	?Q{4e@??a']r@?'eRC[?!M?J??@*	!       2	!       :	?\p@f-?Z*@!gE?D??6@B	!       J	!       R	!       Z	!       b	!       JGPUb q??????!@yc`))?V@