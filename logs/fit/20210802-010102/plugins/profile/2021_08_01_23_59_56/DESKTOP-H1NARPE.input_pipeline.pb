$	???m?Kf@?LC??Gs@̶?ֈ`??!F?6n??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'F?6n??@??fc%f;@1h͏?4e~@I?Z}uU?4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ̶?ֈ`??}?;l"3??1:?`???d?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?wg??????e????1rQ-"??[?r11*?????{@)      0=2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?MbX9??!??jb?tI@)??|?5^??1????3?G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map$(~??k??!iTDt?l?@)??MbX??1???.??6@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??\m????!?`??!!@)??W?2ġ?1!3?% @:Preprocessing2T
Iterator::Root::ParallelMapV2$????ۗ?!?q7???@)$????ۗ?1?q7???@:Preprocessing2E
Iterator::Root??(\?¥?!}(9k>?#@)a2U0*???1P?:T??@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip}?5^?I??!????aM@)?St$????1?̥a??@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?5?;Nё?!`ZU?@)? ?	???1?p?2r@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??_?L??!B???5@)??_?L??1B???5@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?~j?t?h?!$?I?p*??)?~j?t?h?1$?I?p*??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!P?:T????)a2U0*?c?1P?:T????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!?b?????)????Mb`?1?b?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!^?5?AA??)/n??R?1^?5?AA??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI`9"@QT???˸V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	j?/???"@o5?J?U/@}?;l"3??!??fc%f;@	!       "$	r?Cd@?Q?>v?q@rQ-"??[?!h͏?4e~@*	!       2	!       :	)?QG??@R?@??(@!?Z}uU?4@B	!       J	!       R	!       Z	!       b	!       JGPUb q`9"@yT???˸V@