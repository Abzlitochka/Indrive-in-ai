import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import webbrowser
import os

# Загрузка данных
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Очистка данных
    df = df.dropna(subset=['lat', 'lng'])
    df = df[df['lat'].between(50, 52)]  # Фильтр по реальным координатам Алматы
    df = df[df['lng'].between(71, 72)]
    return df

# Создание тепловой карты спроса
def create_demand_heatmap(df, output_file='demand_heatmap.html'):
    # Создаем карту
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=13)
    
    # Подготавливаем данные для тепловой карты
    heat_data = [[row['lat'], row['lng']] for index, row in df.iterrows()]
    
    # Добавляем тепловую карту
    HeatMap(heat_data, radius=15).add_to(m)
    
    # Сохраняем карту
    m.save(output_file)
    return m

# Определение популярных маршрутов
def identify_popular_routes(df, output_file='popular_routes.html'):
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=13)
    
    # Группируем по id поездки
    grouped = df.groupby('randomized_id')
    
    # Создаем маркеры для каждой поездки
    for name, group in grouped:
        if len(group) > 1:  # Только поездки с несколькими точками
            # Создаем линию маршрута
            route_coords = list(zip(group['lat'], group['lng']))
            folium.PolyLine(route_coords, color='blue', weight=2, opacity=0.5).add_to(m)
    
    # Добавляем маркер-кластер для точек
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in df.iterrows():
        folium.Marker([row['lat'], row['lng']]).add_to(marker_cluster)
    
    m.save(output_file)
    return m

# Определение зон с высоким спросом (кластеризация)
def identify_high_demand_zones(df, output_file='high_demand_zones.html', eps=0.001, min_samples=5):
    # Подготовка данных для кластеризации
    coords = df[['lat', 'lng']].values
    
    # DBSCAN кластеризация
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    
    # Добавляем метки кластеров в DataFrame
    df['cluster'] = labels
    
    # Создаем карту
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=13)
    
    # Создаем цветовую схему для кластеров
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive', 'cyan']
    
    # Добавляем кластеры на карту
    for cluster_id in set(labels):
        if cluster_id == -1:  # Шум (не относящийся к кластеру)
            cluster_data = df[df['cluster'] == cluster_id]
            for idx, row in cluster_data.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lng']],
                    radius=3,
                    color='gray',
                    fill=True,
                    fill_color='gray',
                    fill_opacity=0.6
                ).add_to(m)
        else:
            cluster_data = df[df['cluster'] == cluster_id]
            # Определяем центр кластера
            center_lat = cluster_data['lat'].mean()
            center_lng = cluster_data['lng'].mean()
            
            # Определяем размер кластера (количество точек)
            size = len(cluster_data)
            
            # Определяем цвет в зависимости от размера кластера
            color = colors[cluster_id % len(colors)]
            
            # Добавляем круг для кластера
            folium.Circle(
                location=[center_lat, center_lng],
                radius=size * 50,  # Увеличиваем радиус пропорционально количеству точек
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                popup=f"Зона спроса: {size} точек"
            ).add_to(m)
    
    # Добавляем легенду
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 150px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white;
                opacity: 0.8;
                ">
    &nbsp; <b>Легенда</b><br>
    &nbsp; <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> Высокий спрос<br>
    &nbsp; <i style="background: blue; width: 10px; height: 10px; display: inline-block;"></i> Средний спрос<br>
    &nbsp; <i style="background: gray; width: 10px; height: 10px; display: inline-block;"></i> Низкий спрос
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_file)
    return m

# Анализ зон с "проблемами" (низкая скорость, резкие изменения направления)
def identify_problem_zones(df, output_file='problem_zones.html'):
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=13)
    
    # Определяем "проблемные" точки: низкая скорость и резкие изменения направления
    # Для этого вычисляем изменения направления между последовательными точками
    df = df.sort_values(by=['randomized_id', 'lat', 'lng'])
    df['prev_lat'] = df.groupby('randomized_id')['lat'].shift(1)
    df['prev_lng'] = df.groupby('randomized_id')['lng'].shift(1)
    df['prev_azm'] = df.groupby('randomized_id')['azm'].shift(1)
    
    # Вычисляем изменение направления
    df['azm_diff'] = np.abs(df['azm'] - df['prev_azm'])
    df['azm_diff'] = df['azm_diff'].apply(lambda x: min(x, 360-x))  # Учитываем круговую природу азимута
    
    # Выбираем точки с низкой скоростью и резкими изменениями направления
    problem_points = df[(df['spd'] < 5) & (df['azm_diff'] > 30)]
    
    # Добавляем проблемные точки на карту
    for idx, row in problem_points.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.8,
            popup=f"Низкая скорость: {row['spd']}, Резкое изменение направления: {row['azm_diff']}"
        ).add_to(m)
    
    # Добавляем легенду
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 80px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white;
                opacity: 0.8;
                ">
    &nbsp; <b>Проблемные зоны</b><br>
    &nbsp; <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> Низкая скорость и резкие повороты
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_file)
    return m

# Определение "нужных такси зон" (зоны с высоким спросом, но низким количеством водителей)
def identify_taxi_zones(df, output_file='taxi_zones.html'):
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=13)
    
    # Для этого анализа нам нужно предположить, что зоны с высоким спросом - это те, где много точек
    # А зоны с низким количеством водителей - те, где много точек, но мало уникальных id
    # Это грубая оценка, так как у нас нет прямой информации о водителях
    
    # Считаем количество точек и уникальных id в небольших регионах
    # Создаем сетку ячеек
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lng_min, lng_max = df['lng'].min(), df['lng'].max()
    
    lat_step = (lat_max - lat_min) / 10
    lng_step = (lng_max - lng_min) / 10
    
    # Подсчитываем точки в каждой ячейке
    cell_data = []
    for i in range(10):
        for j in range(10):
            lat_start = lat_min + i * lat_step
            lat_end = lat_min + (i+1) * lat_step
            lng_start = lng_min + j * lng_step
            lng_end = lng_min + (j+1) * lng_step
            
            cell_df = df[(df['lat'] >= lat_start) & (df['lat'] < lat_end) & 
                         (df['lng'] >= lng_start) & (df['lng'] < lng_end)]
            
            if len(cell_df) > 0:
                unique_ids = cell_df['randomized_id'].nunique()
                total_points = len(cell_df)
                demand_score = total_points
                supply_score = unique_ids  # Предположим, что количество уникальных id - это количество водителей
                taxi_score = demand_score - supply_score
                
                cell_data.append({
                    'lat': (lat_start + lat_end) / 2,
                    'lng': (lng_start + lng_end) / 2,
                    'demand': demand_score,
                    'supply': supply_score,
                    'taxi_score': taxi_score
                })
    
    # Создаем карту с ячейками
    for cell in cell_data:  # Исправлено: cell_ -> cell_data
        # Определяем цвет в зависимости от taxi_score
        if cell['taxi_score'] > 5:
            color = 'red'
            popup_text = f"Высокий спрос: {cell['demand']}, Низкое предложение: {cell['supply']}"
        elif cell['taxi_score'] > 0:
            color = 'orange'
            popup_text = f"Средний спрос: {cell['demand']}, Среднее предложение: {cell['supply']}"
        else:
            color = 'green'
            popup_text = f"Низкий спрос: {cell['demand']}, Высокое предложение: {cell['supply']}"
        
        folium.Rectangle(
            bounds=[[cell['lat'] - lat_step/2, cell['lng'] - lng_step/2], 
                    [cell['lat'] + lat_step/2, cell['lng'] + lng_step/2]],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.3,
            popup=popup_text
        ).add_to(m)
    
    # Добавляем легенду
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 150px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white;
                opacity: 0.8;
                ">
    &nbsp; <b>Такси зоны</b><br>
    &nbsp; <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> Нужны такси<br>
    &nbsp; <i style="background: orange; width: 10px; height: 10px; display: inline-block;"></i> Средний спрос<br>
    &nbsp; <i style="background: green; width: 10px; height: 10px; display: inline-block;"></i> Достаточно такси
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_file)
    return m

# Основная функция
def main():
    # Загрузка данных
    df = load_data('geotracks.csv')
    
    # Создаем различные визуализации
    print("Создание тепловой карты спроса...")
    create_demand_heatmap(df, 'demand_heatmap.html')
    
    print("Определение популярных маршрутов...")
    identify_popular_routes(df, 'popular_routes.html')
    
    print("Определение зон с высоким спросом...")
    identify_high_demand_zones(df, 'high_demand_zones.html')
    
    print("Анализ проблемных зон...")
    identify_problem_zones(df, 'problem_zones.html')
    
    print("Определение нужных такси зон...")
    identify_taxi_zones(df, 'taxi_zones.html')
    
    print("Все визуализации созданы!")
    
    # Открываем основную карту в браузере
    webbrowser.open('high_demand_zones.html')

if __name__ == "__main__":
    main()