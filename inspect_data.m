load project_data
figure();
subplot(121);
imagesc(imgref);
colormap('gray');
subplot(122);
imagesc(sinogram);
xlabel('views');
ylabel('ray');
