$	?.?9?e@d5!?/?r@???͊?!b?? ?f?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'b?? ?f?@?i?L7=@1#?	 ?}@I2??1@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ???͊?F?Swe??1?'eRC[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!p?71$'??1??zm?I???ْU??r11*	53333?x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap+??	h??!^??d??I@)R???Q??1?!?2/?G@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??镲??!2jK??0@)?8??m4??1??l$??/@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map2U0*???! ?O?v?@)h??|?5??1??l_"?-@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat6?;Nё??!?l$???@)??ݓ????11?????@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip      ??!=?Z5PO@)vq?-??1\???ϩ@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch ?o_Ή?!?8k? @	@) ?o_Ή?1?8k? @	@:Preprocessing2T
Iterator::Root::ParallelMapV2????????!?>?]?	@)????????1?>?]?	@:Preprocessing2E
Iterator::Root??0?*??!f#??@)A??ǘ???1??dԖ>@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeF%u?k?!???s??)F%u?k?1???s??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??_?Le?!???x???)??_?Le?1???x???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!?<?Z5??)????Mb`?1?<?Z5??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!d????)/n??R?1d????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIl?	??!@Q~R???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	@ޫV&|#@?f@?T?0@!?i?L7=@	!       "$	??8?#?c@4???@q@?'eRC[?!#?	 ?}@*	!       2	!       :	?>?c@?1?t2$@!2??1@B	!       J	!       R	!       Z	!       b	!       JGPUb ql?	??!@y~R???V@