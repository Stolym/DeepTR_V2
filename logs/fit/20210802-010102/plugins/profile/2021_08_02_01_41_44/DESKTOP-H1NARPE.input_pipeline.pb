$	?1?Y$?e@z?	?Js@?N^???!??r-?z?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??r-?z?@?YJ???=@1????}@I?4f3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?N^??????????1?'eRC[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!j?@+0d??1Ǻ???v?I|?ʄ_???r11*	??????x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapb??4?8??!???w??L@)9??v????1?.?;-J@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map_?L?J??!e???w?9@)??m4????1?3`??o0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???&??!??ﲻ?"@)?Q?????1??r?!@:Preprocessing2T
Iterator::Root::ParallelMapV2?I+???!f??;F&@)?I+???1f??;F&@:Preprocessing2E
Iterator::Root??|гY??!AhV???$@)????<,??1%?y?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?Q?????!??r?@)K?=?U??1????@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??@??ǈ?!???AM]@)??@??ǈ?1???AM]@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip:#J{?/??!?g?1??O@)?ZӼ?}?1???ϗ??:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?ZӼ?}?!???ϗ??)?ZӼ?}?1???ϗ??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!g??b?T??)a2U0*?c?1g??b?T??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!??/8???)/n??b?1??/8???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!??/8???)/n??R?1??/8???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?'\Ӯ}"@Q{?%J?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????#@??f7!1@!?YJ???=@	!       "$	M?P}??c@?!_q?Eq@?'eRC[?!????}@*	!       2	!       :	???r_q@?E?!?%@!?4f3@B	!       J	!       R	!       Z	!       b	!       JGPUb q?'\Ӯ}"@y{?%J?V@