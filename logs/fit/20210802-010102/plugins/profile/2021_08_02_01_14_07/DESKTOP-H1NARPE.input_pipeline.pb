$	?<?AQ@???*q2c@iUMu?!?H?{u@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?H?{u@?#???=@1qr?Cks@Ik'JB"	@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails 'jin????<???ܴ??1iUMu_?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsiUMu?1iUMu?r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?t><K???1?t><K???r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?I?U??1?3?ۃ`?I??hUM??r11*	gffff?y@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap:#J{?/??!???A??J@)??H.???1G@C???H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?C?l????!?:?p?8@)46<???1?t?@??1@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?V-??!X?1е,!@)? ?	???1CB??8?@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatB>?٬???!$?T?a@)	?c???1?3??r@:Preprocessing2T
Iterator::Root::ParallelMapV2???<,Ԛ?!q?{??Y@)???<,Ԛ?1q?{??Y@:Preprocessing2E
Iterator::RootEGr????!5?z`ח&@)ˡE?????1?z??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??ZӼ???!^????@)??ZӼ???1^????@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip~??k	???!??}?	P@)??~j?t??1瞯rVb@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?g??s?u?!4H?4H???)?g??s?u?14H?4H???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU???N@s?!?I??0??)U???N@s?1?I??0??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????Mb`?!?A5?v???)????Mb`?1?A5?v???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6Z?!К*?+???)-C??6Z?1К*?+???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??Vi?#@Q?"?R??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	bO;?5?@c#???*@!?#???=@	!       "$	?G??9O@pÉ?/^a@iUMu_?!qr?Cks@*	!       2	!       :	?2?p)??2?gv????!k'JB"	@B	!       J	!       R	!       Z	!       b	!       JGPUb q??Vi?#@y?"?R??V@