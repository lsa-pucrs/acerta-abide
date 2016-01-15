$(document).ready(function(){

  $('.btn-group button').click(function(){
    var fold = $(this).val();
    $('.btn-group button').removeClass('active');
    $(this).addClass('active');
    var query = {'fold': fold};
    $.getJSON('data/' + JSON.stringify(query), function(json){
      $('#graph').highcharts({
        chart: { type: 'bar' },
        title: { text: '' },
        xAxis: { categories: ['train', 'valid', 'test'] },
        yAxis: { min: 0, },
        legend: { reversed: true },
        plotOptions: {
          series: {
            stacking: 'percent'
          }
        },
        series: json.data
      });
    });
  });

  $('.btn-group button:first').click();

});