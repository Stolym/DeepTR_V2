$	????8?e@??m???r@??!????!Yk(?n?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'Yk(?n?@???Y.;=@1Q???&?}@I??	14@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??!????6w??\???1??????`?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!@?3iS???1<??~Ka?I}<?ݭ???r11*	?????az@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????ׁ??!?E???tI@) ?o_???1HT?n?G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??B?i???!)?DO?@@)Dio??ɴ?1?an??<3@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?<,Ԛ???!????ѫ+@)??HP??1?F[?3*@:Preprocessing2T
Iterator::Root::ParallelMapV2??JY?8??!ԷNX?@)??JY?8??1ԷNX?@:Preprocessing2E
Iterator::Root?g??s???!?=??$@)Q?|a2??1i?2???@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat"??u????!?k??L@)V-???1???L{@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??y?):??!??"?7? @)??y?):??1??"?7? @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipA??ǘ???!?Ǟ??pL@)????Mb??1O?^?S??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea??+ei?!????d???)a??+ei?1????d???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?????g?!s&~`=???)?????g?1s&~`=???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOf?!c?l?x??)??_vOf?1c?l?x??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!??&Q???)?~j?t?X?1??&Q???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?
ȷ.#@Q??)=?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????~#@T??s??0@!???Y.;=@	!       "$	`݋B%?c@#& ?51q@??????`?!Q???&?}@*	!       2	!       :	X??e@?gM?&@!??	14@B	!       J	!       R	!       Z	!       b	!       JGPUb q?
ȷ.#@y??)=?V@