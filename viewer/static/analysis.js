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
    if(this._highlight.indexOf(region) > -1){
      return this._cat_color(region);
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

    $('#weights .btn-group button').click(function(){
      var fold = $(this).val();
      $('#weights .btn-group button').removeClass('active');
      $(this).addClass('active');
      $.getJSON('analysis/weights/' + fold, function(weights) {

        $('#weights .list-group a').remove();

        for(var i = 0; i < 5; i++){

          var feature = weights.asd[i][0];
          var weight = weights.asd[i][1];
          var conn = analysis_view.connections[feature];
          var elem = $('<a href="#" class="list-group-item" data-connection="'+conn+'" data-feature="'+feature+'">'+
                    '<h4 class="list-group-item-heading"></h4>'+
                    '<p class="list-group-item-text"></p>'+
                    '</a>');
          elem.find('h4').html('Corr ' + conn);
          elem.find('p').html(weight);
          $('#weights .list-group.asd').append(elem);


          var feature = weights.tc[i][0];
          var weight = weights.tc[i][1];
          var conn = analysis_view.connections[feature];
          var elem = $('<a href="#" class="list-group-item" data-connection="'+conn+'" data-feature="'+feature+'">'+
                    '<h4 class="list-group-item-heading"></h4>'+
                    '<p class="list-group-item-text"></p>'+
                    '</a>');
          elem.find('h4').html('Corr ' + conn);
          elem.find('p').html(weight);
          $('#weights .list-group.tc').append(elem);

        }

        $('#weights .list-group a').click(function(){
          var connection = $(this).data('connection');
          $('#weights .list-group a').removeClass('active');
          $('#weights .list-group a').filter(function(){
            return $(this).data('connection') == connection;
          }).addClass('active');
          $.getJSON('analysis/distribution/' + $(this).data('feature'), function(data) {
            plot_distribution(connection, data);
            var conns = connection.split(',');
            brain_view.highlight([parseInt(conns[0]), parseInt(conns[1])]);
          });
          return false;
        });

      });
    });

    $('#weights .btn-group button.mean').click();

  });
});

function plot_distribution(title, values){
  values.asd.unshift(0);
  values.tc.unshift(1);
  $('#distribution').highcharts({
    chart: { type: 'boxplot', inverted: true },
    title: { text: 'Distribution of ' + title },
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
}