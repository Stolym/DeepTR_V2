$	?H?sj?q@???ݿ?x@?lscz¢?!Ժj??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'Ժj??@
i?A'H?@1???b??@I?????1@r0"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?lscz¢?Tb.???1rQ-"??[?r11*?????x@)      0=2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapS?!?uq??!y?bɍ?K@)???K7???1?????I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map[B>?٬??!? ?:;@)c?ZB>???1???H?C*@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatS?!?uq??!y?bɍ?+@)B`??"۩?1?W?w?6*@:Preprocessing2T
Iterator::Root::ParallelMapV2a2U0*???!OVa???@)a2U0*???1OVa???@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??ZӼ???!?k?=?-@)??y?):??1??Ħz@:Preprocessing2E
Iterator::RootQ?|a??!??8??"@)?o_???1???NV@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipmV}??b??!Ƿz???O@)??_?L??1?T???@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???_vO~?!e??????)???_vO~?1e??????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicea??+ey?!{????)a??+ey?1{????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea??+ei?!{????)a??+ei?1{????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!?T?????)??_?Le?1?T?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!?T?????)??_?LU?1?T?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIN??~!@Q?<??'?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??XQ/@B?;CK6@Tb.???!
i?A'H?@	!       "$	1ZGU??o@p???G?v@rQ-"??[?!???b??@*	!       2	!       :	?????!@~?(?m)@!?????1@B	!       J	!       R	!       Z	!       b	!       JGPUb qN??~!@y?<??'?V@