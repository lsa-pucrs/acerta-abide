var NeuroView = function(jQuery, elem, dims, params){

  this.$ = jQuery;

  this.elem = jQuery(elem);

  this.x = this.y = this.z = 0;
  this.size = params.size || 3;
  this.cross = params.cross_size || 5;
  if(params.cross_size === false)
    this.cross = false
  this.dims = dims;
  this.params = params;

  this._loaded = false;
  this._load_callbacks = [];
  this._change_callbacks = [];
  this._highlight_callbacks = [];

}

NeuroView.prototype.init = function(voxels){

  var $nv = this;

  $nv.shape = shape = {
    x: voxels.length, y: voxels[0].length, z: voxels[0][0].length
  };

  $nv.min = +Infinity;
  $nv.max = -Infinity;

  for (var x = 0; x < shape.x; x++) {
    voxels[x].reverse();
    for (var y = 0; y < shape.y; y++){
      voxels[x][y].reverse();
      for (var z = 0; z < shape.z; z++){
        $nv.max = max($nv.max, voxels[x][y][z]);
        $nv.min = min($nv.min, voxels[x][y][z]);
      }
    }
  }

  $nv.voxels = voxels;
  $nv._highlight = [];

  $nv._color_linear = d3.scale.linear().domain([$nv.min, $nv.max]).range([0, 200]);
  $nv._color_highlight = function(r){ return 255; };

  $nv.onhighlight(function(regions){
    $nv.draw();
  });

  $nv.onchange(function(oldp, newp){
    var dims = '';
    if(oldp == undefined){
      dims = 'xyz';
    } else {
      if(oldp.x != newp.x) dims += 'x';
      if(oldp.y != newp.y) dims += 'y';
      if(oldp.z != newp.z) dims += 'z';
    }
    for(var d in $nv.dims){
      var d = $nv.dims[d];
      var update = false;
      if(dims.indexOf($nv.missing(d)) == -1)
        update = [oldp, newp];
      $nv.drawView(d, update);
    }
  });

  $nv.elem[0].innerHTML = '';

  for(var d in $nv.dims){
    var dims = $nv.dims[d];
    var c = $nv.canvas(dims, true)
      .data('dim', dims)
      .attr('width', shape[dims[0]] * $nv.size)
      .attr('height', shape[dims[1]] * $nv.size)
      .on('mousedown mousemove', function(e){
        e.stopPropagation();
        e.preventDefault();
        if(e.buttons == 1){
          var $t = $(this);
          var off = $t.offset();
          var x = Math.floor( (e.pageX - off.left) / $nv.size );
          var y = Math.floor( (e.pageY - off.top) / $nv.size );
          var dims = $t.data('dim');
          if(!$nv.params.highlight || !e.ctrlKey){
            var p = $nv.pos();
            p[dims[0]] = x;
            p[dims[1]] = y;
            $nv.change(p);
          }
        }
      })
      .on('mousedown', function(e){
        e.stopPropagation();
        e.preventDefault();
        if($nv.params.highlight && e.buttons == 1 && e.ctrlKey){
          var $t = $(this);
          var off = $t.offset();
          var x = Math.floor( (e.pageX - off.left) / $nv.size );
          var y = Math.floor( (e.pageY - off.top) / $nv.size );
          var dims = $t.data('dim');
          var p = $nv.pos();
          p[dims[0]] = x;
          p[dims[1]] = y;
          var val = $nv.val(p);
          var contains = $nv._highlight.indexOf(val);
          if(contains > -1){
            $nv._highlight.splice(contains, 1);
          } else {
            if(e.shiftKey && $nv.params.highlight == 'multi'){
              $nv._highlight.push(val);
            } else {
              $nv._highlight = [$nv.val(p)];
            }
          }
          $nv.trigger_highlight();
        }
      });
  }

  $nv.change({
    x: Math.round(shape.x/2),
    y: Math.round(shape.y/2),
    z: Math.round(shape.z/2),
  });

  $nv.trigger_load();

};

NeuroView.prototype.contourn = function(){ return false; };

NeuroView.prototype.color = function(region){
  if(this._highlight.indexOf(region) > -1){
    var c = this._color_highlight(region);
  } else {
    var c = this._color_linear(region);
  }
  return d3.rgb(c, c, c).toString();
};

NeuroView.prototype.trigger_load = function(){
  this._loaded = true;
  for(var i in this._load_callbacks){
    this._load_callbacks[i]();
  }
};

NeuroView.prototype.onload = function(callback){
  if(this._loaded){
    callback();
  } else {
    this._load_callbacks.push(callback);
  }
};

NeuroView.prototype.trigger_change = function(oldp, newp){
  for(var i in this._change_callbacks){
    this._change_callbacks[i](oldp, newp);
  }
};

NeuroView.prototype.onchange = function(callback){
  this._change_callbacks.push(callback);
};

NeuroView.prototype.change = function(newp){
  var oldp = this.pos();
  this.x = newp.x;
  this.y = newp.y;
  this.z = newp.z;
  this.trigger_change(oldp, newp);
};

NeuroView.prototype.trigger_highlight = function(){
  for(var i in this._highlight_callbacks){
    this._highlight_callbacks[i](this._highlight.slice());
  }
};

NeuroView.prototype.onhighlight = function(callback){
  this._highlight_callbacks.push(callback);
};

NeuroView.prototype.highlight = function(regions){
  if(regions != undefined){
    this._highlight = regions;
    this.trigger_highlight();
  }
  return this._highlight;
};

NeuroView.prototype.val = function(point){
  var point = this.limit(point);
  return this.voxels[point.x][point.y][point.z];
};

NeuroView.prototype.limit = function(p){
  p.x = min(max(0, p.x), this.shape.x-1);
  p.y = min(max(0, p.y), this.shape.y-1);
  p.z = min(max(0, p.z), this.shape.z-1);
  return p;
};

NeuroView.prototype.pos = function(x, y, z){
  if(x != undefined){
    return this.limit({ x: x, y: y, z: z });
  }
  return this.limit({ x: this.x, y: this.y, z: this.z });
};

NeuroView.prototype.dim = function(d){
  return ['x', 'y', 'z'].indexOf(d);
}

NeuroView.prototype.missing = function(dims){
  return ['x', 'y', 'z'].filter(function(n) {
    return dims.indexOf(n) == -1
  })[0];
}

NeuroView.prototype.canvas = function(dims, jq){
  jq = typeof jq !== 'undefined' ? jq : false;
  var o = this.$('.' + dims, this.$(this.elem));
  if(o.size() == 0){
    this.$(this.elem).append(this.$('<canvas class="' + dims + '" />'));
    return this.canvas(dims, jq);
  }
  return jq ? o : o[0];
}

NeuroView.prototype.context = function(dims, clear){
  clear = typeof clear !== 'undefined' ? clear : false;
  var c = this.canvas(dims);
  var ctx = c.getContext('2d');
  if(clear)
    ctx.clearRect(0, 0, c.width, c.height);
  return ctx;
}

NeuroView.prototype.outline = function(matrix){
  var shape = [matrix.length, matrix[0].length];
  var contourn = zeros(shape[0], shape[1]);
  for (var i = 0; i < shape[0]; i++) {
    var min = shape[1], max = -1;
    for (var j = 0; j < shape[1]; j++) {
      if(matrix[i][j] == 1){
        if(j < min) min = j;
        if(j > max) max = j;
      }
    }
    if(min < shape[1]) contourn[i][min] = 1;
    if(max > -1) contourn[i][max] = 1;
  }

  for (var i = 0; i < shape[1]; i++) {
    var min = shape[0], max = -1;
    for (var j = 0; j < shape[0]; j++) {
      if(matrix[j][i] == 1){
        if(j < min) min = j;
        if(j > max) max = j;
      }
    }
    if(min < shape[0]) contourn[min][i] = 1;
    if(max > -1) contourn[max][i] = 1;
  }

  return contourn;
}

NeuroView.prototype.fillView = function(dims){
  var dim1 = dims[0], dim2 = dims[1];
  var fill = zeros(this.shape[dim1], this.shape[dim2]);
  for (var x = 0; x < this.shape.x; x++) {
    for (var y = 0; y < this.shape.y; y++) {
      for (var z = 0; z < this.shape.z; z++) {
        var p = this.pos(x, y, z);
        if(this.val(p) > 0){
          fill[p[dim1]][p[dim2]] = 1;
        }
      }
    }
  }
  return this.outline(fill);
}

NeuroView.prototype.paintContourn = function(dims){
  if(this.outline == undefined)
    this.outline = {};
  if(this.outline[dims] == undefined)
    this.outline[dims] = this.fillView(dims);
  var fill = this.outline[dims];
  var ctx = this.context(dims);
  var dim1 = dims[0], dim2 = dims[1];
  for (var i = 0; i < this.shape[dim1]; i++) {
    for (var j = 0; j < this.shape[dim2]; j++) {
      if(fill[i][j] == 1){
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(i * this.size,  j * this.size,
                     this.size, this.size);
      }
    }
  }
}

NeuroView.prototype.drawCrossAt = function(ctx, x, y){
  if(!this.cross)
    return;
  ctx.fillStyle = '#337ab7';
  for (var i = -2; i <= 2; i++)
    ctx.fillRect((x+i) * this.size, y * this.size, this.size, this.size);
  for (var i = -2; i <= 2; i++)
    ctx.fillRect(x * this.size, (y+i) * this.size, this.size, this.size);
}

NeuroView.prototype.clearCross = function(ctx, dims, old){
  if(!this.cross)
    return;
  var size = Math.floor(this.cross / 2);
  var dim1 = dims[0], dim2 = dims[1];
  for (var i = -size; i <= size; i++){
    for (var j = -size; j <= size; j++){
      var p = Object.create(old);
      p[dim1] += i;
      p[dim2] += j;
      crossarm = this.cross / 2;
      var voxel = this.val(p);
      ctx.fillStyle = this.color(voxel);
      ctx.fillRect(p[dim1] * this.size, p[dim2] * this.size,
                   this.size, this.size);
    }
  }
}

NeuroView.prototype.drawDimensions = function(dims){
  var ctx = this.context(dims, false);
  var dim1 = dims[0], dim2 = dims[1];
  var voxel = this.pos();
  for (var i = 0; i < this.shape[dim1]; i++) {
    for (var j = 0; j < this.shape[dim2]; j++) {
      voxel[dim1] = i;
      voxel[dim2] = j;
      ctx.fillStyle = this.color(this.val(voxel));
      ctx.fillRect( i * this.size, j * this.size,
                    this.size, this.size);
    }
  }
}

NeuroView.prototype.drawView = function(dims, update){
  var ctx = this.context(dims, !update);
  if(update)
    this.clearCross(ctx, dims, update[0]);
  else
    this.drawDimensions(dims);
  if(this.contourn())
    this.paintContourn(dims);
  this.drawCrossAt(ctx, this[dims[0]], this[dims[1]]);
  this.canvas(dims, true).trigger('render');
}

NeuroView.prototype.drawViews = function(){
  for(var d in this.dims)
    this.drawView(this.dims[d], false);
}

NeuroView.prototype.draw = function(){
  this.drawViews();
}

NeuroView.prototype.centerAt = function(region){
  if(!region.hasOwnProperty('length'))
    region = [region];
  var c = this.findCenterOf(region);
  this.x = c.x;
  this.y = c.y;
  this.z = c.z;
  this.draw();
  return c;
}

NeuroView.prototype.findCenterOf = function(r){
  var min = [this.shape.x, this.shape.y, this.shape.z];
  var max = [0, 0, 0];
  for (var x = 0; x < this.shape.x; x++) {
    for (var y = 0; y < this.shape.y; y++) {
      for (var z = 0; z < this.shape.z; z++) {
        if(r.indexOf(this.voxels[x][y][z]) > -1){
          if(x < min[0]) min[0] = x;
          if(x > max[0]) max[0] = x;
          if(y < min[1]) min[1] = y;
          if(y > max[1]) max[1] = y;
          if(z < min[2]) min[2] = z;
          if(z > max[2]) max[2] = z;
        }
      }
    }
  }
  return {
    x: Math.round((max[0] + min[0]) / 2),
    y: Math.round((max[1] + min[1]) / 2),
    z: Math.round((max[2] + min[2]) / 2),
  };
}

function zeros(rows, cols) {
  var array = [], row = [];
  while (cols--) row.push(0);
  while (rows--) array.push(row.slice());
  return array;
}

min = Math.min
max = Math.max