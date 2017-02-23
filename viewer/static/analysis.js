window.analysis_view = {};

$(document).ready(function () {

  $('#distribution').highcharts({
    chart: { type: 'boxplot', inverted: true },
    title: { text: 'Distribution' },
    legend: { enabled: false },
    xAxis: {
      categories: ['ASD', 'TC'],
      title: { text: 'Class' }
    },
    yAxis: {
      title: { text: false },
      max: 1, min: -1,
    },
    series: [
      { name: 'ASD', data: [], },
      { name: 'TC', data: [], },
    ]
  });

  window.brain_view = new NeuroView($, $('#brain')[0] , ['xy', 'yz', 'xz'], {
    highlight: false,
    size: 3,
    cross_size: false,
  });

  brain_view._cat_color = d3.scale.category20b();
  brain_view.color = function(region){
    if(this._highlight.length == 0){
      var c = this._color_linear(region);
      return d3.rgb(c, c, c).toString();
    }
    var i = this._highlight.indexOf(region);
    if(i > -1){
      if(i == 0){
        return d3.rgb('green').toString();
      } else {
        color = d3.scale.linear().domain([1, this._highlight.length]).range([d3.rgb('blue'), d3.rgb('blue').darker(3)]);
        return color(i).toString();
      }
    } else {
      var c = 0;
    }
    return d3.rgb(c, c, c).toString();
  };

  brain_view.contourn = function(){
    if(this._highlight.length == 0)
      return false;
    return true;
  };

  brain_view._drawDimensions = brain_view.drawDimensions;
  brain_view.drawDimensions = function(dims){
    if(this._highlight.length == 0)
      return brain_view._drawDimensions(dims);

    var ctx = this.context(dims, false);
    var dim1 = dims[0], dim2 = dims[1];
    var fill = zeros(this.shape[dim1], this.shape[dim2]);

    for (var x = 0; x < this.shape.x; x++) {
      for (var y = 0; y < this.shape.y; y++) {
        for (var z = 0; z < this.shape.z; z++) {
          var p = this.pos(x, y, z);
          var region = this.val(p);
          if(this._highlight.indexOf(region) > -1)
            fill[p[dim1]][p[dim2]] = this.val(p);
        }
      }
    }
    for (var i = 0; i < this.shape[dim1]; i++) {
      for (var j = 0; j < this.shape[dim2]; j++) {
        if(fill[i][j] > 0){
          ctx.fillStyle = this.color(fill[i][j]);
          ctx.fillRect( i * this.size, j * this.size,
                        this.size, this.size);
        }
      }
    }
  };

  $.getJSON('analysis/mask', function(data) {
    brain_view.init(data.voxels);
    brain_view.draw();
    analysis_view.connections = data.connections;
    init_buttons();
  });

  $('#features table').hide();

  $('#features .nav li').click(function(){
    $('#features .nav li').removeClass('active');
    $('#features table').hide();
    $(this).addClass('active');
    $( $(this).data('elem') ).show();
    return false;
  }).filter(':first').click();

});

function load_svm(fold, callback){
  $.getJSON('analysis/svm/' + fold, function(data) {
    analysis_view.svm = data.svm;
    callback();
  });
}

function load_atlas(callback){
  $.getJSON('analysis/atlas', function(data) {
    analysis_view.atlas = data;
    callback();
  });
}

function corrformat(conn){
  var i = conn.split(',');
  return i[0] + ' ↔ ' + i[1];
}

function unique(arr){
   var u = {}, a = [];
   for(var i = 0, l = arr.length; i < l; ++i){
      if(u.hasOwnProperty(arr[i])) {
         continue;
      }
      a.push(arr[i]);
      u[arr[i]] = 1;
   }
   return a;
}

function corrlabels(conn){
  var conns = conn.split(',');
  $('#from h3').html('Parcel ' + conns[0]);
  $('#from').show();
  parcellabels(conns[0], $('#from .list-group'));

  if(conns.length == 2){
    $('#distribution').show();
    $('#to h3').html('Parcel ' + conns[1]);
    $('#to').show();
    parcellabels(conns[1], $('#to .list-group'));
  } else {
    $('#distribution').hide();
    $('#to h3').html('connects with');
    $('#to').show();
    var connectswith = unique(analysis_view.connectswith[conns[0]]);

    var highlight = [parseInt(conns[0])];

    $('#to .list-group').empty();
    for(var i in connectswith){

      highlight.push(parseInt(connectswith[i]));

      var from = Math.min(conns[0], connectswith[i]);
      var to = Math.max(conns[0], connectswith[i]);

      var el = $('<div class="panel panel-default clickable" data-connection="'+ from + ',' + to +'"><div class="panel-body">'+
          '<h4 class="pull-left">' + connectswith[i] + '</h4>'+
      '</div></div>');

      el.click(function(){
        var connection = $(this).data('connection');
        connectionview(connection);
      });

      $('#to .list-group').append(el);
    }

    brain_view.highlight(highlight);

  }
}

function parcellabels(parcel, elem){
  elem.empty();
  for(var atlas in analysis_view.atlas[parcel]){
    for(var i in analysis_view.atlas[parcel][atlas]){
      if(analysis_view.atlas[parcel][atlas][i] == "None")
        continue;
      elem.append('<div class="panel panel-default"><div class="panel-body">'+
          '<h4 class="pull-left">' + analysis_view.atlas[parcel][atlas][i] + '</h4>'+
          '<p class="pull-right">' + atlas + '</p>'+
      '</div></div>')
    }
  }
}

function FilledArray(len, val) {
    var rv = new Array(len);
    while (--len >= 0) {
        rv[len] = val;
    }
    return rv;
}

function sortkeys(dict){
  var items = Object.keys(dict).map(function(key) {
    return [key, dict[key]];
  });
  items.sort(function(first, second) {
    return second[1] - first[1];
  });
  return items.map(function(val) {
    return val[0];
  });
}

function connectionview(connection){
    corrlabels(connection);
    plot_distribution(connection);
    var conns = connection.split(',');
    brain_view.highlight([parseInt(conns[0]), parseInt(conns[1])]);
}

function load_features(fold){
  $.getJSON('analysis/weights/first.valid.final.mlp-valid.' + fold, function(weights) {

    $('#relevant tbody').empty();

    analysis_view.weights = weights;
    analysis_view.overall = {
      asd: {},
      tc: {}
    };

    analysis_view.connectswith = {

    };

    for(var i = 0; i < 40; i++){

      var feature = weights.asd[i][0];
      var weight = weights.asd[i][1];
      var conn = analysis_view.connections[feature];
      var parcels = conn.split(',')
      analysis_view.overall.asd[parcels[0]] = ( analysis_view.overall.asd[parcels[0]] | 0 ) + 1;
      analysis_view.overall.asd[parcels[1]] = ( analysis_view.overall.asd[parcels[1]] | 0 ) + 1;

      if(!(parcels[0] in analysis_view.connectswith))
        analysis_view.connectswith[parcels[0]] = [];
      analysis_view.connectswith[parcels[0]].push(parcels[1]);
      if(!(parcels[1] in analysis_view.connectswith))
        analysis_view.connectswith[parcels[1]] = [];
      analysis_view.connectswith[parcels[1]].push(parcels[0]);

      var feature = weights.tc[i][0];
      var weight = weights.tc[i][1];
      var conn = analysis_view.connections[feature];
      var parcels = conn.split(',')
      analysis_view.overall.tc[parcels[0]] = ( analysis_view.overall.tc[parcels[0]] | 0 ) + 1;
      analysis_view.overall.tc[parcels[1]] = ( analysis_view.overall.tc[parcels[1]] | 0 ) + 1;

      if(!(parcels[0] in analysis_view.connectswith))
        analysis_view.connectswith[parcels[0]] = [];
      analysis_view.connectswith[parcels[0]].push(parcels[1]);
      if(!(parcels[1] in analysis_view.connectswith))
        analysis_view.connectswith[parcels[1]] = [];
      analysis_view.connectswith[parcels[1]].push(parcels[0]);

      var tr = $('<tr />');

      var elem = $('<td />');
      elem.html(i+1);
      tr.append(elem);

      var feature = weights.asd[i][0];
      var weight = weights.asd[i][1];
      var conn = analysis_view.connections[feature];
      var elem = $('<td class="clickable" data-connection="'+conn+'" data-feature="'+feature+'" />');
      elem.html('<h4></h4>'+'<p class="svm"></p>'+'<p class="w"></p>');
      elem.find('h4').html(corrformat(conn));
      elem.find('p.svm').html('SVM: ' + (analysis_view.svm.indexOf(feature) + 1) + ' &ordm;');
      elem.find('p.w').html('Colab: ' + (weight * 100).toFixed(2) + ' %');
      tr.append(elem);

      var feature = weights.tc[i][0];
      var weight = weights.tc[i][1];
      var conn = analysis_view.connections[feature];
      var elem = $('<td class="clickable" data-connection="'+conn+'" data-feature="'+feature+'" />');
      elem.html('<h4></h4>'+'<p class="svm"></p>'+'<p class="w"></p>');
      elem.find('h4').html(corrformat(conn));
      elem.find('p.svm').html('SVM: ' + (analysis_view.svm.indexOf(feature) + 1) + ' &ordm;');
      elem.find('p.w').html('Colab: ' + (weight * 100).toFixed(2) + ' %');
      tr.append(elem);

      // var feature = analysis_view.svm[i];
      // var conn = analysis_view.connections[feature];
      // var elem = $('<td class="clickable" data-connection="'+conn+'" data-feature="'+feature+'" />');
      // elem.html('<h4></h4>');
      // elem.find('h4').html(corrformat(conn));
      // tr.append(elem);

      $('#relevant tbody').append(tr);

    }

    $('#relevant td.clickable').click(function(){
      var connection = $(this).data('connection');
      $('#relevant td.clickable').removeClass('active');
      $('#relevant td.clickable').filter(function(){
        return $(this).data('connection') == connection;
      }).addClass('active');

      connectionview(connection);

      return false;
    });

    $('#recurrent tbody').empty();

    analysis_view.overall_order = {
      asd: sortkeys(analysis_view.overall.asd),
      tc: sortkeys(analysis_view.overall.tc),
    }

    var len = Math.min(analysis_view.overall_order.asd.length, analysis_view.overall_order.tc.length);
    for(var i = 0; i < len; i++){
      var tr = $('<tr />');

      var elem = $('<td />');
      elem.html(i+1);
      tr.append(elem);

      var parcel = analysis_view.overall_order.asd[i];
      var elem = $('<td class="clickable" data-parcel="'+parcel+'" />');
      elem.html('<h4></h4>'+'<p class="n"></p>');
      elem.find('h4').html(parcel);
      elem.find('p.n').html('Ocurrences: ' + analysis_view.overall.asd[parcel]);
      tr.append(elem);

      var parcel = analysis_view.overall_order.tc[i];
      var elem = $('<td class="clickable" data-parcel="'+parcel+'" />');
      elem.html('<h4></h4>'+'<p class="n"></p>');
      elem.find('h4').html(parcel);
      elem.find('p.n').html('Ocurrences: ' + analysis_view.overall.tc[parcel]);
      tr.append(elem);

      $('#recurrent tbody').append(tr);
    }

    $('#recurrent td.clickable').click(function(){
      var parcel = $(this).data('parcel') + "";
      $('#recurrent td.clickable').removeClass('active');
      $('#recurrent td.clickable').filter(function(){
        return $(this).data('parcel') == parcel;
      }).addClass('active');
      corrlabels(parcel);
      return false;
    });

  });
}

function init_buttons(){
  $('#folds .btn-group button').click(function(){
    var fold = $(this).val();
    $('#folds .btn-group button').removeClass('active');
    $(this).addClass('active');

    $('#relevant tbody').empty();
    $('#recurrent tbody').empty();

    load_atlas(function(){
      load_svm(fold, function(){
        load_features(fold);
      });
    });

  });
  $('#folds .btn-group button').filter(function(){ return $(this).val() == 'mean'; }).click();
}

function plot_distribution(connection){
  $.getJSON('analysis/distribution/' + connection, function(values) {

    values.asd.unshift(0);
    values.tc.unshift(1);

    $('#distribution').highcharts({
      chart: { type: 'boxplot', inverted: true },
      title: { text: 'Distribution of ' + corrformat(connection) },
      legend: { enabled: false },
      xAxis: {
        categories: ['ASD', 'TC'],
        title: { text: 'Class' }
      },
      yAxis: {
        title: { text: false },
        max: 1, min: -1,
      },
      series: [
        { name: 'ASD', data: [values.asd], },
        { name: 'TC', data: [values.tc], },
      ]
    });

  });
}