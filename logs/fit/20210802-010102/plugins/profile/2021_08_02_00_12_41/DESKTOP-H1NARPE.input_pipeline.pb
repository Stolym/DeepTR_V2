$	?
?a7g@t????t@?&?|?W?!]??h?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails']??h?@?`?Y=@1?????@I???tt7@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?&?|?W?1?&?|?W?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??o{?Ĳ?1!>???@p?I?4}v???r11*	43333?u@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapGr?????!]??J@)??/?$??170?~M?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?A?fշ?!&T?$X?:@)??H.?!??1????o0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?:pΈ??!*L???$@)??镲??1?$m??<#@:Preprocessing2T
Iterator::Root::ParallelMapV2?~j?t???!?g?mɺ@)?~j?t???1?g?mɺ@:Preprocessing2E
Iterator::RootbX9?Ȧ?!?Ř?*?)@)ˡE?????1Y#???@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!??!}?????@) ?o_Ή?1iSD??@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipl	??g???!?$C>? N@)??ǘ????1{?+]??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?<,Ԛ?}?!?D?m? @)?<,Ԛ?}?1?D?m? @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicea??+ei?!6Z
?i???)a??+ei?16Z
?i???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?????g?!mu")???)?????g?1mu")???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!???+H??)??_?Le?1???+H??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceŏ1w-!_?!}???????)ŏ1w-!_?1}???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI G'???"@Q;???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????&?#@??c?0@!?`?Y=@	!       "$	Q\h?e@???Ta2r@?&?|?W?!?????@*	!       2	!       :	???]@?????
+@!???tt7@B	!       J	!       R	!       Z	!       b	!       JGPUb q G'???"@y;???V@