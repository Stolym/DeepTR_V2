$	??K8??f@v???l?s@d?]K???!?Ɍ???@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?Ɍ???@߉Y/??;@1???(@Ih??`o?4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails d?]K???]??'???1$D??b?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?9?????1???%f?I??v1?t??r11*	??????v@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapŏ1w-!??!]?(?u?H@)>?٬?\??1o?$??G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map
ףp=
??!?:Fq??8@)?HP???1??	??*@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?0?*??!?|٠?&@)??ʡE???1>???SK%@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?p=
ף??!Q????!@)???_vO??1h?e?&_ @:Preprocessing2T
Iterator::Root::ParallelMapV2A??ǘ???!??Ź?@)A??ǘ???1??Ź?@:Preprocessing2E
Iterator::Root?g??s???!?8?1?s'@)M?O???1????ZX@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-!?lV??!d?:F?O@)????Mb??1?/4??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchΈ?????!4??A?@)Έ?????14??A?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice_?Q?k?!{ja????)_?Q?k?1{ja????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!??ZX????)?????g?1??ZX????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!M?l????)??_vOf?1M?l????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!"??x??)/n??R?1"??x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIH?n??!@Qw>"???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?2??"@]T?֝0@!߉Y/??;@	!       "$	c?????d@?<HpF?q@$D??b?!???(@*	!       2	!       :	?????@????'@!h??`o?4@B	!       J	!       R	!       Z	!       b	!       JGPUb qH?n??!@yw>"???V@