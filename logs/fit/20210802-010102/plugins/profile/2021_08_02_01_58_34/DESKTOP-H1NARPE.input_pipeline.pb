$	?z?;g@?5Q?s@F_A??h??!?\??vJ?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?\??vJ?@??_?|?=@1??>?X@I%?c\q96@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ???*ø??V?F?????1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!F_A??h????U????1rQ-"??[?r11*	53333?x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapı.n???!??????K@)a??+e??1,F?Na?I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapꕲq???!9?S?H;@)B>?٬???1$I?$I?,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??|?5^??!M???9?)@)???JY???1??J.(@:Preprocessing2E
Iterator::Root*??Dؠ?!??/?? @)?J?4??1^H??@:Preprocessing2T
Iterator::Root::ParallelMapV2??ܵ?|??!??Ox!A@)??ܵ?|??1??Ox!A@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat6?;Nё??!T?HQ?*@)?q??????1?ag`?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?(??0??!? *B ?@)?(??0??1? *B ?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchn????!?q??@)n????1?q??@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?|a2U??!??\P@)?? ?rh??1G???)@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??H?}m?!?2?l???)??H?}m?1?2?l???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceF%u?k?!?x8???)F%u?k?1?x8???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??H?}]?!?2?l???)??H?}]?1?2?l???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIptY??"@Qrє??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??ć??#@??^?1@??U????!??_?|?=@	!       "$	%m??d@??д?r@rQ-"??[?!??>?X@*	!       2	!       :	ܒ/{??@?jI??)@!%?c\q96@B	!       J	!       R	!       Z	!       b	!       JGPUb qptY??"@yrє??V@