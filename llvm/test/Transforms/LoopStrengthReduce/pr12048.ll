; RUN: opt < %s -loop-reduce

define void @resolve_name(i1 %arg, i8 %arg2) nounwind uwtable ssp {
  br label %while.cond40.preheader
while.cond132.while.cond.loopexit_crit_edge:
  br label %while.cond40.preheader
while.cond40.preheader:
  br label %while.cond40
while.cond40:
  %indvars.iv194 = phi ptr [ null, %while.cond40.preheader ], [ %scevgep, %while.body51 ]
  %tmp.1 = phi ptr [ undef, %while.cond40.preheader ], [ %incdec.ptr, %while.body51 ]
  switch i8 %arg2, label %while.body51 [
    i8 0, label %if.then59
  ]
while.body51:                                     ; preds = %land.end50
  %incdec.ptr = getelementptr inbounds i8, ptr %tmp.1, i64 1
  %scevgep = getelementptr i8, ptr %indvars.iv194, i64 1
  br label %while.cond40
if.then59:                                        ; preds = %while.end
  br i1 %arg, label %if.then64, label %if.end113
if.then64:                                        ; preds = %if.then59
  %incdec.ptr88.tmp.2 = select i1 undef, ptr undef, ptr undef
  br label %if.end113
if.end113:                                        ; preds = %if.then64, %if.then59
  %tmp.4 = phi ptr [ %incdec.ptr88.tmp.2, %if.then64 ], [ undef, %if.then59 ]
  %tmp.4195 = ptrtoint ptr %tmp.4 to i64
  br  label %while.cond132.preheader
while.cond132.preheader:                          ; preds = %if.end113
  %cmp133173 = icmp eq ptr %tmp.1, %tmp.4
  br i1 %cmp133173, label %while.cond40.preheader, label %while.body139.lr.ph
while.body139.lr.ph:                              ; preds = %while.cond132.preheader
  %scevgep198199 = ptrtoint ptr %indvars.iv194 to i64
  br label %while.body139
while.body139:                                    ; preds = %while.body139, %while.body139.lr.ph
  %start_of_var.0177 = phi ptr [ %tmp.1, %while.body139.lr.ph ], [ null, %while.body139 ]
  br i1 %arg, label %while.cond132.while.cond.loopexit_crit_edge, label %while.body139
}
