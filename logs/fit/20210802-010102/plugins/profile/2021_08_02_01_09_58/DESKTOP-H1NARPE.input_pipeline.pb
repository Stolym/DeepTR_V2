$	7<??U1f@ߔ?L?7s@?h㈵?T?!`?U,???@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'`?U,???@{??>"<@1࢓?Vq~@I???>\1@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?h㈵?T?1?h㈵?T?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?>?֧̔?1?o??}u?I>]ݱ?&??r11*	53333?x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???߾??!?]>̟?K@)1?*????1?i?I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map|??Pk???!?ÓD?L7@)?U???د?1S??p/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???Q???!+??uS@)??Ɯ?1? ???g@:Preprocessing2T
Iterator::Root::ParallelMapV2Ǻ?????!ct???@)Ǻ?????1ct???@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatJ+???!hLw(?@)Ǻ?????1ct???@:Preprocessing2E
Iterator::Root??ܥ?!x??a??%@)Dio??ɔ?1?yq?E?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???{????!D??"%zP@)???????1a?`?+I@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchǺ?????!ct???@)Ǻ?????1ct???@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??ZӼ?t?!????&???)??ZӼ?t?1????&???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!VJ8????)?J?4a?1VJ8????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeŏ1w-!_?!:?M????)ŏ1w-!_?1:?M????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!rr??????)/n??R?1rr??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIPW??)!@QU????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?XL?"@?pP?7>0@!{??>"<@	!       "$	=???Kd@?Y??t?q@?h㈵?T?!࢓?Vq~@*	!       2	!       :	W?o??3@H???$@!???>\1@B	!       J	!       R	!       Z	!       b	!       JGPUb qPW??)!@yU????V@